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
| nccl-async | 0.048719 | 0.9194 | +0.0069 | 1924.9 | 1075 | 41.2 | 100% | 100% | 4.8 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9194 | nccl-async | - | - |

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
| nccl-async | 1.9831 | 0.6713 | 0.5228 | 0.5366 | 0.5043 | 0.4895 | 0.4706 | 0.4146 | 0.3922 | 0.3798 | 0.1931 | 0.1536 | 0.1328 | 0.1181 | 0.1188 | 0.0621 | 0.0567 | 0.0517 | 0.0512 | 0.0487 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4058 | 2.7 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.2992 | 3.2 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2950 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 404 | 389 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1923.7 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu2 | 1923.8 | 1.1 | epoch-boundary(199) |
| nccl-async | gpu0 | 1923.7 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.6s |
| resnet-graph | nccl-async | gpu1 | 1.2s | 0.0s | 0.0s | 0.0s | 2.0s |
| resnet-graph | nccl-async | gpu2 | 1.1s | 0.0s | 0.0s | 0.0s | 1.1s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 503 | 0 | 1075 | 41.2 | 649/7119 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 170.4 | 8.9% |

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
| resnet-graph | nccl-async | 195 | 1075 | 0 | 7.02e-3 | +1.84e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 1075 | 6.67e-2 | 4.71e-2 | 0.00e0 | 4.23e-1 | 53.1 | -1.13e-4 | 3.72e-3 |
| resnet-graph | nccl-async | 1 | 1075 | 6.69e-2 | 4.90e-2 | 0.00e0 | 3.87e-1 | 28.9 | -1.18e-4 | 5.71e-3 |
| resnet-graph | nccl-async | 2 | 1075 | 6.61e-2 | 4.85e-2 | 0.00e0 | 3.74e-1 | 18.0 | -1.16e-4 | 5.64e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9859 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9863 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9972 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 73 (0,1,2,3,4,5,6,7…136,144) | 2 (14,22) | 14 | 0,1,2,3,4,5,6,7…136,144 | 22 |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 56 | 56 |
| resnet-graph | nccl-async | 0e0 | 5 | 32 | 32 |
| resnet-graph | nccl-async | 0e0 | 10 | 12 | 12 |
| resnet-graph | nccl-async | 1e-4 | 3 | 38 | 38 |
| resnet-graph | nccl-async | 1e-4 | 5 | 22 | 22 |
| resnet-graph | nccl-async | 1e-4 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-3 | 3 | 2 | 2 |
| resnet-graph | nccl-async | 1e-3 | 5 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 781 | +0.038 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 71 | +0.168 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 218 | +0.030 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 1072 | +0.012 | 194 | +0.085 | +0.053 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 1073 | 3.35e1–8.15e1 | 6.75e1 | 1.42e-3 | 2.82e-3 | 3.70e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 783 | 63–77984 | +4.238e-7 | 0.001 | +4.439e-7 | 0.001 | 100 | +6.572e-7 | 0.001 | 29–760 | +2.085e-3 | 0.536 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 764 | 912–77984 | +1.530e-7 | 0.000 | +1.526e-7 | 0.000 | 99 | +2.400e-7 | 0.000 | 32–760 | +2.077e-3 | 0.568 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 72 | 78733–117130 | +9.235e-6 | 0.094 | +9.048e-6 | 0.090 | 48 | +6.033e-6 | 0.030 | 361–831 | -7.499e-4 | 0.062 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 219 | 117582–155588 | -1.664e-5 | 0.037 | -1.715e-5 | 0.038 | 47 | +1.158e-5 | 0.034 | 37–1006 | +2.176e-3 | 0.589 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +2.085e-3 | r0: +2.060e-3, r1: +2.094e-3, r2: +2.110e-3 | r0: 0.601, r1: 0.497, r2: 0.503 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +2.077e-3 | r0: +2.053e-3, r1: +2.086e-3, r2: +2.101e-3 | r0: 0.644, r1: 0.524, r2: 0.529 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -7.499e-4 | r0: -7.057e-4, r1: -7.764e-4, r2: -7.664e-4 | r0: 0.056, r1: 0.067, r2: 0.065 | 1.10× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +2.176e-3 | r0: +2.146e-3, r1: +2.184e-3, r2: +2.206e-3 | r0: 0.614, r1: 0.576, r2: 0.569 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇█▇▇███████████████████▆▅▅▅▅▅▆▆▆▆▆▆▃▂▂▂▂▂▂▁▃▃▃▃` | `▂▄▅▆▆█▅▆▆▅▅▅▅▅▅▅▅▄▄▆▄▅▄▅▄▃▃▄▄▅▅▅▅▅▅▅▅▁▃▄▅▃▃▃▆█▇▆▆` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 20 | 0.00e0 | 4.23e-1 | 8.03e-2 | 5.74e-2 | 19 | -7.65e-2 | +2.03e-2 | -9.34e-3 | -3.35e-3 |
| 1 | 3.00e-1 | 10 | 6.04e-2 | 1.21e-1 | 7.72e-2 | 9.14e-2 | 30 | -3.78e-2 | +4.08e-2 | +1.47e-3 | -1.37e-4 |
| 2 | 3.00e-1 | 10 | 8.53e-2 | 1.20e-1 | 9.44e-2 | 9.28e-2 | 22 | -1.09e-2 | +1.09e-2 | +1.31e-4 | -2.68e-5 |
| 3 | 3.00e-1 | 13 | 6.67e-2 | 1.33e-1 | 8.13e-2 | 8.25e-2 | 22 | -4.52e-2 | +2.98e-2 | -7.24e-4 | -2.13e-4 |
| 4 | 3.00e-1 | 15 | 7.80e-2 | 1.41e-1 | 8.98e-2 | 8.16e-2 | 22 | -2.97e-2 | +1.99e-2 | -3.52e-4 | -4.48e-4 |
| 5 | 3.00e-1 | 11 | 7.30e-2 | 1.40e-1 | 8.60e-2 | 8.22e-2 | 20 | -3.48e-2 | +2.45e-2 | -3.68e-4 | -3.51e-4 |
| 6 | 3.00e-1 | 14 | 6.30e-2 | 1.34e-1 | 8.17e-2 | 8.30e-2 | 20 | -2.43e-2 | +2.80e-2 | -7.93e-5 | +3.80e-5 |
| 7 | 3.00e-1 | 13 | 6.55e-2 | 1.40e-1 | 8.58e-2 | 8.77e-2 | 20 | -4.76e-2 | +3.42e-2 | -4.71e-4 | -3.12e-4 |
| 8 | 3.00e-1 | 12 | 7.54e-2 | 1.44e-1 | 8.48e-2 | 7.75e-2 | 19 | -3.22e-2 | +3.14e-2 | -1.80e-4 | -3.88e-4 |
| 9 | 3.00e-1 | 22 | 5.68e-2 | 1.40e-1 | 7.16e-2 | 6.83e-2 | 13 | -4.81e-2 | +3.53e-2 | -4.48e-4 | +1.80e-5 |
| 10 | 3.00e-1 | 10 | 6.07e-2 | 1.38e-1 | 7.97e-2 | 7.75e-2 | 17 | -5.15e-2 | +5.46e-2 | +1.42e-3 | +5.06e-4 |
| 11 | 3.00e-1 | 14 | 6.58e-2 | 1.38e-1 | 7.61e-2 | 7.38e-2 | 17 | -3.91e-2 | +4.15e-2 | +4.88e-4 | +3.38e-4 |
| 12 | 3.00e-1 | 20 | 5.54e-2 | 1.35e-1 | 7.02e-2 | 7.43e-2 | 18 | -5.94e-2 | +5.49e-2 | +5.44e-4 | +5.61e-4 |
| 13 | 3.00e-1 | 12 | 6.85e-2 | 1.37e-1 | 7.70e-2 | 7.68e-2 | 15 | -3.72e-2 | +3.13e-2 | +6.25e-5 | +4.39e-4 |
| 14 | 3.00e-1 | 15 | 6.18e-2 | 1.53e-1 | 8.26e-2 | 9.34e-2 | 31 | -5.34e-2 | +5.72e-2 | +9.37e-4 | +6.05e-4 |
| 15 | 3.00e-1 | 7 | 7.89e-2 | 1.52e-1 | 9.57e-2 | 7.89e-2 | 19 | -2.06e-2 | +1.35e-2 | -1.85e-3 | -7.58e-4 |
| 16 | 3.00e-1 | 20 | 5.84e-2 | 1.31e-1 | 7.04e-2 | 7.19e-2 | 13 | -3.59e-2 | +3.39e-2 | +7.98e-5 | +5.43e-4 |
| 17 | 3.00e-1 | 11 | 5.88e-2 | 1.37e-1 | 7.67e-2 | 6.78e-2 | 20 | -4.08e-2 | +4.97e-2 | +5.75e-4 | +9.66e-5 |
| 18 | 3.00e-1 | 16 | 6.94e-2 | 1.40e-1 | 8.08e-2 | 8.56e-2 | 24 | -3.55e-2 | +3.41e-2 | +5.19e-4 | +4.60e-4 |
| 19 | 3.00e-1 | 9 | 6.28e-2 | 1.38e-1 | 8.17e-2 | 7.98e-2 | 19 | -2.57e-2 | +1.79e-2 | -1.14e-3 | -1.76e-4 |
| 20 | 3.00e-1 | 16 | 5.77e-2 | 1.35e-1 | 7.04e-2 | 8.14e-2 | 19 | -4.61e-2 | +3.44e-2 | -1.78e-4 | +7.53e-4 |
| 21 | 3.00e-1 | 16 | 5.57e-2 | 1.48e-1 | 7.15e-2 | 8.10e-2 | 13 | -5.34e-2 | +3.86e-2 | +8.96e-6 | +1.37e-3 |
| 22 | 3.00e-1 | 16 | 5.29e-2 | 1.38e-1 | 6.81e-2 | 6.38e-2 | 16 | -6.56e-2 | +6.81e-2 | +6.88e-4 | +3.84e-4 |
| 23 | 3.00e-1 | 14 | 5.85e-2 | 1.45e-1 | 7.18e-2 | 7.89e-2 | 43 | -5.56e-2 | +5.26e-2 | +7.05e-4 | +8.84e-4 |
| 24 | 3.00e-1 | 9 | 8.63e-2 | 1.55e-1 | 1.07e-1 | 9.34e-2 | 24 | -1.39e-2 | +6.30e-3 | -5.31e-4 | -4.45e-5 |
| 25 | 3.00e-1 | 15 | 7.01e-2 | 1.39e-1 | 8.18e-2 | 7.56e-2 | 20 | -2.75e-2 | +2.35e-2 | -4.23e-4 | -3.22e-4 |
| 26 | 3.00e-1 | 2 | 7.82e-2 | 8.07e-2 | 7.94e-2 | 8.07e-2 | 22 | +1.45e-3 | +1.63e-3 | +1.54e-3 | +3.09e-5 |
| 27 | 3.00e-1 | 2 | 8.47e-2 | 2.38e-1 | 1.62e-1 | 2.38e-1 | 214 | +1.92e-4 | +4.84e-3 | +2.51e-3 | +5.26e-4 |
| 28 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 210 | -7.13e-4 | -7.13e-4 | -7.13e-4 | +4.02e-4 |
| 29 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 210 | -4.14e-5 | -4.14e-5 | -4.14e-5 | +3.58e-4 |
| 30 | 3.00e-1 | 2 | 1.98e-1 | 1.99e-1 | 1.98e-1 | 1.99e-1 | 178 | -1.39e-4 | +2.88e-5 | -5.49e-5 | +2.80e-4 |
| 31 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 220 | -3.75e-4 | -3.75e-4 | -3.75e-4 | +2.15e-4 |
| 32 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 202 | +4.39e-4 | +4.39e-4 | +4.39e-4 | +2.37e-4 |
| 33 | 3.00e-1 | 2 | 1.93e-1 | 1.94e-1 | 1.93e-1 | 1.94e-1 | 178 | -1.88e-4 | +2.59e-5 | -8.12e-5 | +1.78e-4 |
| 34 | 3.00e-1 | 2 | 1.85e-1 | 1.95e-1 | 1.90e-1 | 1.95e-1 | 178 | -2.40e-4 | +2.99e-4 | +2.94e-5 | +1.52e-4 |
| 35 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 190 | -2.92e-4 | -2.92e-4 | -2.92e-4 | +1.08e-4 |
| 36 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 196 | +1.76e-4 | +1.76e-4 | +1.76e-4 | +1.15e-4 |
| 37 | 3.00e-1 | 2 | 1.92e-1 | 1.96e-1 | 1.94e-1 | 1.96e-1 | 178 | +3.68e-5 | +1.17e-4 | +7.69e-5 | +1.08e-4 |
| 38 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 187 | -2.45e-4 | -2.45e-4 | -2.45e-4 | +7.26e-5 |
| 39 | 3.00e-1 | 2 | 1.90e-1 | 1.91e-1 | 1.90e-1 | 1.90e-1 | 180 | -3.70e-5 | +9.71e-5 | +3.00e-5 | +6.38e-5 |
| 40 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 177 | -1.14e-4 | -1.14e-4 | -1.14e-4 | +4.60e-5 |
| 41 | 3.00e-1 | 2 | 1.84e-1 | 1.99e-1 | 1.92e-1 | 1.99e-1 | 144 | -5.42e-5 | +5.68e-4 | +2.57e-4 | +8.92e-5 |
| 42 | 3.00e-1 | 2 | 1.70e-1 | 1.75e-1 | 1.72e-1 | 1.75e-1 | 144 | -1.02e-3 | +2.06e-4 | -4.05e-4 | +1.47e-6 |
| 43 | 3.00e-1 | 2 | 1.74e-1 | 1.89e-1 | 1.82e-1 | 1.89e-1 | 144 | -1.25e-5 | +5.65e-4 | +2.76e-4 | +5.65e-5 |
| 44 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 174 | -5.50e-4 | -5.50e-4 | -5.50e-4 | -4.10e-6 |
| 45 | 3.00e-1 | 2 | 1.87e-1 | 1.89e-1 | 1.88e-1 | 1.89e-1 | 156 | +7.00e-5 | +4.46e-4 | +2.58e-4 | +4.38e-5 |
| 46 | 3.00e-1 | 2 | 1.80e-1 | 1.85e-1 | 1.83e-1 | 1.85e-1 | 144 | -2.76e-4 | +1.84e-4 | -4.59e-5 | +2.90e-5 |
| 47 | 3.00e-1 | 2 | 1.74e-1 | 1.80e-1 | 1.77e-1 | 1.80e-1 | 144 | -3.90e-4 | +2.18e-4 | -8.61e-5 | +1.02e-5 |
| 48 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 155 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -4.50e-6 |
| 49 | 3.00e-1 | 3 | 1.70e-1 | 1.84e-1 | 1.78e-1 | 1.70e-1 | 136 | -6.07e-4 | +1.82e-4 | -1.02e-4 | -3.79e-5 |
| 50 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 166 | +3.02e-6 | +3.02e-6 | +3.02e-6 | -3.38e-5 |
| 51 | 3.00e-1 | 2 | 1.87e-1 | 1.89e-1 | 1.88e-1 | 1.89e-1 | 129 | +4.98e-5 | +5.26e-4 | +2.88e-4 | +2.49e-5 |
| 52 | 3.00e-1 | 2 | 1.69e-1 | 1.80e-1 | 1.75e-1 | 1.80e-1 | 126 | -7.04e-4 | +5.12e-4 | -9.58e-5 | +8.08e-6 |
| 53 | 3.00e-1 | 2 | 1.68e-1 | 1.86e-1 | 1.77e-1 | 1.86e-1 | 118 | -4.27e-4 | +8.85e-4 | +2.29e-4 | +5.66e-5 |
| 54 | 3.00e-1 | 2 | 1.64e-1 | 1.79e-1 | 1.71e-1 | 1.79e-1 | 124 | -8.20e-4 | +7.31e-4 | -4.46e-5 | +4.52e-5 |
| 55 | 3.00e-1 | 2 | 1.65e-1 | 1.80e-1 | 1.72e-1 | 1.80e-1 | 118 | -5.14e-4 | +7.10e-4 | +9.82e-5 | +6.14e-5 |
| 56 | 3.00e-1 | 2 | 1.65e-1 | 1.83e-1 | 1.74e-1 | 1.83e-1 | 118 | -5.30e-4 | +8.78e-4 | +1.74e-4 | +8.98e-5 |
| 57 | 3.00e-1 | 2 | 1.64e-1 | 1.83e-1 | 1.74e-1 | 1.83e-1 | 118 | -6.69e-4 | +8.99e-4 | +1.15e-4 | +1.02e-4 |
| 58 | 3.00e-1 | 3 | 1.55e-1 | 1.78e-1 | 1.66e-1 | 1.55e-1 | 105 | -1.32e-3 | +7.07e-4 | -4.26e-4 | -4.74e-5 |
| 59 | 3.00e-1 | 2 | 1.60e-1 | 1.82e-1 | 1.71e-1 | 1.82e-1 | 109 | +2.11e-4 | +1.15e-3 | +6.80e-4 | +9.55e-5 |
| 60 | 3.00e-1 | 3 | 1.53e-1 | 1.69e-1 | 1.61e-1 | 1.53e-1 | 96 | -1.04e-3 | +5.41e-4 | -4.85e-4 | -6.32e-5 |
| 61 | 3.00e-1 | 2 | 1.55e-1 | 1.68e-1 | 1.62e-1 | 1.68e-1 | 91 | +1.15e-4 | +9.16e-4 | +5.15e-4 | +5.07e-5 |
| 62 | 3.00e-1 | 3 | 1.48e-1 | 1.68e-1 | 1.57e-1 | 1.54e-1 | 91 | -1.04e-3 | +1.34e-3 | -2.15e-4 | -2.13e-5 |
| 63 | 3.00e-1 | 3 | 1.52e-1 | 1.73e-1 | 1.59e-1 | 1.52e-1 | 91 | -1.46e-3 | +1.36e-3 | -5.13e-5 | -4.34e-5 |
| 64 | 3.00e-1 | 3 | 1.54e-1 | 1.68e-1 | 1.59e-1 | 1.54e-1 | 91 | -9.36e-4 | +8.53e-4 | +3.81e-5 | -3.25e-5 |
| 65 | 3.00e-1 | 2 | 1.49e-1 | 1.66e-1 | 1.58e-1 | 1.66e-1 | 89 | -2.94e-4 | +1.25e-3 | +4.76e-4 | +7.19e-5 |
| 66 | 3.00e-1 | 4 | 1.40e-1 | 1.78e-1 | 1.52e-1 | 1.42e-1 | 80 | -3.07e-3 | +2.30e-3 | -3.45e-4 | -8.04e-5 |
| 67 | 3.00e-1 | 4 | 1.39e-1 | 1.69e-1 | 1.50e-1 | 1.43e-1 | 79 | -1.39e-3 | +2.47e-3 | +2.58e-5 | -6.94e-5 |
| 68 | 3.00e-1 | 2 | 1.43e-1 | 1.67e-1 | 1.55e-1 | 1.67e-1 | 67 | +3.54e-5 | +2.27e-3 | +1.16e-3 | +1.74e-4 |
| 69 | 3.00e-1 | 5 | 1.26e-1 | 1.56e-1 | 1.36e-1 | 1.33e-1 | 73 | -3.16e-3 | +2.23e-3 | -5.10e-4 | -7.84e-5 |
| 70 | 3.00e-1 | 3 | 1.30e-1 | 1.61e-1 | 1.43e-1 | 1.30e-1 | 54 | -3.99e-3 | +2.51e-3 | -3.68e-4 | -2.00e-4 |
| 71 | 3.00e-1 | 6 | 1.09e-1 | 1.59e-1 | 1.21e-1 | 1.09e-1 | 45 | -6.93e-3 | +5.61e-3 | -5.18e-4 | -3.89e-4 |
| 72 | 3.00e-1 | 5 | 1.08e-1 | 1.53e-1 | 1.20e-1 | 1.12e-1 | 40 | -7.52e-3 | +5.97e-3 | -1.29e-4 | -3.31e-4 |
| 73 | 3.00e-1 | 8 | 8.93e-2 | 1.45e-1 | 1.07e-1 | 8.93e-2 | 30 | -1.09e-2 | +7.10e-3 | -9.39e-4 | -8.04e-4 |
| 74 | 3.00e-1 | 10 | 7.10e-2 | 1.50e-1 | 8.84e-2 | 8.01e-2 | 24 | -1.91e-2 | +1.60e-2 | -4.02e-4 | -4.75e-4 |
| 75 | 3.00e-1 | 13 | 6.13e-2 | 1.49e-1 | 7.74e-2 | 8.57e-2 | 19 | -4.63e-2 | +3.48e-2 | +1.63e-4 | +5.29e-4 |
| 76 | 3.00e-1 | 12 | 5.93e-2 | 1.48e-1 | 8.36e-2 | 8.64e-2 | 20 | -5.02e-2 | +4.12e-2 | -9.29e-5 | +1.46e-4 |
| 77 | 3.00e-1 | 14 | 6.28e-2 | 1.41e-1 | 7.49e-2 | 7.26e-2 | 15 | -4.21e-2 | +3.67e-2 | -1.81e-4 | -4.14e-5 |
| 78 | 3.00e-1 | 16 | 5.38e-2 | 1.45e-1 | 7.02e-2 | 7.16e-2 | 18 | -6.83e-2 | +5.70e-2 | -7.67e-5 | +1.98e-4 |
| 79 | 3.00e-1 | 14 | 5.51e-2 | 1.51e-1 | 7.66e-2 | 8.28e-2 | 18 | -5.68e-2 | +4.74e-2 | +3.52e-4 | +6.06e-4 |
| 80 | 3.00e-1 | 13 | 5.65e-2 | 1.50e-1 | 8.05e-2 | 8.01e-2 | 19 | -4.46e-2 | +4.37e-2 | +3.75e-4 | +2.87e-4 |
| 81 | 3.00e-1 | 14 | 7.07e-2 | 1.48e-1 | 7.97e-2 | 7.07e-2 | 18 | -3.70e-2 | +3.54e-2 | -3.50e-4 | -4.24e-4 |
| 82 | 3.00e-1 | 16 | 6.34e-2 | 1.39e-1 | 7.96e-2 | 8.10e-2 | 23 | -3.67e-2 | +3.66e-2 | +2.91e-4 | +1.05e-4 |
| 83 | 3.00e-1 | 11 | 5.63e-2 | 1.63e-1 | 7.87e-2 | 7.12e-2 | 18 | -7.34e-2 | +3.96e-2 | -1.84e-3 | -8.83e-4 |
| 84 | 3.00e-1 | 19 | 6.33e-2 | 1.53e-1 | 7.54e-2 | 8.12e-2 | 23 | -4.47e-2 | +4.16e-2 | +5.24e-4 | +5.38e-4 |
| 85 | 3.00e-1 | 8 | 7.27e-2 | 1.51e-1 | 8.83e-2 | 7.75e-2 | 20 | -2.93e-2 | +2.31e-2 | -1.10e-3 | -4.40e-4 |
| 86 | 3.00e-1 | 14 | 6.77e-2 | 1.55e-1 | 7.80e-2 | 7.79e-2 | 18 | -3.15e-2 | +2.93e-2 | -3.66e-4 | +3.78e-5 |
| 87 | 3.00e-1 | 15 | 5.56e-2 | 1.45e-1 | 7.16e-2 | 6.72e-2 | 18 | -6.82e-2 | +4.95e-2 | -6.56e-4 | -3.92e-4 |
| 88 | 3.00e-1 | 15 | 5.01e-2 | 1.53e-1 | 7.57e-2 | 7.55e-2 | 19 | -7.37e-2 | +5.31e-2 | -1.24e-3 | -5.30e-4 |
| 89 | 3.00e-1 | 15 | 5.98e-2 | 1.44e-1 | 7.42e-2 | 6.73e-2 | 17 | -5.08e-2 | +4.09e-2 | -7.41e-4 | -6.48e-4 |
| 90 | 3.00e-1 | 16 | 5.74e-2 | 1.56e-1 | 7.04e-2 | 6.63e-2 | 17 | -5.34e-2 | +5.08e-2 | -1.12e-4 | -1.27e-4 |
| 91 | 3.00e-1 | 14 | 6.58e-2 | 1.50e-1 | 7.71e-2 | 7.11e-2 | 17 | -4.18e-2 | +3.86e-2 | -2.06e-5 | -1.30e-4 |
| 92 | 3.00e-1 | 11 | 7.27e-2 | 1.55e-1 | 9.18e-2 | 9.28e-2 | 23 | -2.36e-2 | +3.17e-2 | +8.66e-4 | +4.23e-4 |
| 93 | 3.00e-1 | 20 | 5.83e-2 | 1.57e-1 | 7.47e-2 | 8.45e-2 | 18 | -3.06e-2 | +2.89e-2 | -2.36e-5 | +9.71e-4 |
| 94 | 3.00e-1 | 10 | 5.46e-2 | 1.36e-1 | 7.25e-2 | 6.68e-2 | 15 | -4.39e-2 | +3.98e-2 | -2.26e-3 | -8.50e-4 |
| 95 | 3.00e-1 | 16 | 5.68e-2 | 1.40e-1 | 7.84e-2 | 7.12e-2 | 17 | -5.71e-2 | +5.14e-2 | -1.08e-4 | -7.17e-4 |
| 96 | 3.00e-1 | 2 | 6.52e-2 | 6.91e-2 | 6.72e-2 | 6.91e-2 | 17 | -4.89e-3 | +3.49e-3 | -7.02e-4 | -6.73e-4 |
| 97 | 3.00e-1 | 2 | 6.43e-2 | 2.53e-1 | 1.59e-1 | 2.53e-1 | 233 | -2.85e-4 | +5.88e-3 | +2.80e-3 | +1.74e-5 |
| 98 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 267 | -5.26e-4 | -5.26e-4 | -5.26e-4 | -3.70e-5 |
| 99 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 263 | +1.47e-4 | +1.47e-4 | +1.47e-4 | -1.86e-5 |
| 100 | 3.00e-2 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 270 | -1.43e-4 | -1.43e-4 | -1.43e-4 | -3.10e-5 |
| 101 | 3.00e-2 | 1 | 1.56e-1 | 1.56e-1 | 1.56e-1 | 1.56e-1 | 237 | -1.46e-3 | -1.46e-3 | -1.46e-3 | -1.74e-4 |
| 102 | 3.00e-2 | 1 | 2.09e-2 | 2.09e-2 | 2.09e-2 | 2.09e-2 | 257 | -7.81e-3 | -7.81e-3 | -7.81e-3 | -9.38e-4 |
| 103 | 3.00e-2 | 1 | 2.36e-2 | 2.36e-2 | 2.36e-2 | 2.36e-2 | 248 | +4.86e-4 | +4.86e-4 | +4.86e-4 | -7.96e-4 |
| 104 | 3.00e-2 | 1 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 265 | +3.70e-5 | +3.70e-5 | +3.70e-5 | -7.12e-4 |
| 105 | 3.00e-2 | 2 | 2.53e-2 | 2.68e-2 | 2.61e-2 | 2.68e-2 | 202 | +2.24e-4 | +2.96e-4 | +2.60e-4 | -5.27e-4 |
| 107 | 3.00e-2 | 2 | 2.43e-2 | 2.75e-2 | 2.59e-2 | 2.75e-2 | 202 | -3.62e-4 | +6.12e-4 | +1.25e-4 | -3.98e-4 |
| 108 | 3.00e-2 | 1 | 2.54e-2 | 2.54e-2 | 2.54e-2 | 2.54e-2 | 236 | -3.37e-4 | -3.37e-4 | -3.37e-4 | -3.92e-4 |
| 109 | 3.00e-2 | 1 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 227 | +3.05e-4 | +3.05e-4 | +3.05e-4 | -3.22e-4 |
| 110 | 3.00e-2 | 2 | 2.78e-2 | 2.84e-2 | 2.81e-2 | 2.84e-2 | 202 | +9.51e-5 | +9.54e-5 | +9.53e-5 | -2.43e-4 |
| 112 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 326 | +2.15e-6 | +2.15e-6 | +2.15e-6 | -2.19e-4 |
| 113 | 3.00e-2 | 2 | 3.32e-2 | 3.38e-2 | 3.35e-2 | 3.32e-2 | 180 | -1.05e-4 | +6.26e-4 | +2.60e-4 | -1.31e-4 |
| 114 | 3.00e-2 | 1 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 202 | -7.05e-4 | -7.05e-4 | -7.05e-4 | -1.89e-4 |
| 115 | 3.00e-2 | 2 | 3.02e-2 | 3.07e-2 | 3.04e-2 | 3.07e-2 | 180 | +8.80e-5 | +2.31e-4 | +1.60e-4 | -1.23e-4 |
| 116 | 3.00e-2 | 1 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 234 | -1.11e-4 | -1.11e-4 | -1.11e-4 | -1.22e-4 |
| 117 | 3.00e-2 | 1 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 222 | +5.60e-4 | +5.60e-4 | +5.60e-4 | -5.37e-5 |
| 118 | 3.00e-2 | 1 | 3.31e-2 | 3.31e-2 | 3.31e-2 | 3.31e-2 | 231 | -9.18e-5 | -9.18e-5 | -9.18e-5 | -5.75e-5 |
| 119 | 3.00e-2 | 2 | 3.43e-2 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 180 | +1.24e-5 | +1.55e-4 | +8.38e-5 | -3.13e-5 |
| 120 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 221 | -2.45e-4 | -2.45e-4 | -2.45e-4 | -5.27e-5 |
| 121 | 3.00e-2 | 1 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 249 | +2.86e-4 | +2.86e-4 | +2.86e-4 | -1.88e-5 |
| 122 | 3.00e-2 | 2 | 3.72e-2 | 3.73e-2 | 3.72e-2 | 3.73e-2 | 168 | +1.98e-6 | +2.63e-4 | +1.33e-4 | +8.70e-6 |
| 123 | 3.00e-2 | 1 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 206 | -6.33e-4 | -6.33e-4 | -6.33e-4 | -5.55e-5 |
| 124 | 3.00e-2 | 2 | 3.66e-2 | 3.73e-2 | 3.70e-2 | 3.73e-2 | 168 | +1.06e-4 | +5.53e-4 | +3.29e-4 | +1.54e-5 |
| 125 | 3.00e-2 | 1 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 205 | -3.63e-4 | -3.63e-4 | -3.63e-4 | -2.25e-5 |
| 126 | 3.00e-2 | 1 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 231 | +3.74e-4 | +3.74e-4 | +3.74e-4 | +1.72e-5 |
| 127 | 3.00e-2 | 2 | 3.96e-2 | 4.02e-2 | 3.99e-2 | 3.96e-2 | 188 | -7.98e-5 | +3.02e-4 | +1.11e-4 | +3.31e-5 |
| 128 | 3.00e-2 | 1 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 260 | -2.36e-4 | -2.36e-4 | -2.36e-4 | +6.23e-6 |
| 129 | 3.00e-2 | 1 | 4.41e-2 | 4.41e-2 | 4.41e-2 | 4.41e-2 | 226 | +7.54e-4 | +7.54e-4 | +7.54e-4 | +8.11e-5 |
| 130 | 3.00e-2 | 2 | 4.06e-2 | 4.34e-2 | 4.20e-2 | 4.06e-2 | 162 | -3.99e-4 | -8.99e-5 | -2.45e-4 | +1.77e-5 |
| 131 | 3.00e-2 | 2 | 3.74e-2 | 4.02e-2 | 3.88e-2 | 4.02e-2 | 162 | -4.51e-4 | +4.45e-4 | -3.10e-6 | +1.82e-5 |
| 132 | 3.00e-2 | 1 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 201 | -1.60e-4 | -1.60e-4 | -1.60e-4 | +3.45e-7 |
| 133 | 3.00e-2 | 2 | 4.27e-2 | 4.38e-2 | 4.32e-2 | 4.38e-2 | 162 | +1.58e-4 | +4.45e-4 | +3.01e-4 | +5.61e-5 |
| 134 | 3.00e-2 | 1 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 193 | -4.48e-4 | -4.48e-4 | -4.48e-4 | +5.64e-6 |
| 135 | 3.00e-2 | 1 | 4.37e-2 | 4.37e-2 | 4.37e-2 | 4.37e-2 | 213 | +3.95e-4 | +3.95e-4 | +3.95e-4 | +4.46e-5 |
| 136 | 3.00e-2 | 2 | 4.58e-2 | 4.63e-2 | 4.60e-2 | 4.63e-2 | 145 | +7.58e-5 | +2.33e-4 | +1.54e-4 | +6.46e-5 |
| 137 | 3.00e-2 | 2 | 3.86e-2 | 4.45e-2 | 4.16e-2 | 4.45e-2 | 137 | -9.96e-4 | +1.04e-3 | +2.47e-5 | +6.73e-5 |
| 138 | 3.00e-2 | 2 | 3.95e-2 | 4.50e-2 | 4.23e-2 | 4.50e-2 | 137 | -6.57e-4 | +9.47e-4 | +1.45e-4 | +9.00e-5 |
| 139 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 185 | -7.71e-4 | -7.71e-4 | -7.71e-4 | +3.88e-6 |
| 140 | 3.00e-2 | 2 | 4.53e-2 | 4.57e-2 | 4.55e-2 | 4.53e-2 | 145 | -7.24e-5 | +9.08e-4 | +4.18e-4 | +7.76e-5 |
| 141 | 3.00e-2 | 2 | 4.07e-2 | 4.73e-2 | 4.40e-2 | 4.73e-2 | 137 | -5.99e-4 | +1.09e-3 | +2.47e-4 | +1.18e-4 |
| 142 | 3.00e-2 | 2 | 4.10e-2 | 4.61e-2 | 4.35e-2 | 4.61e-2 | 137 | -8.79e-4 | +8.55e-4 | -1.19e-5 | +1.02e-4 |
| 143 | 3.00e-2 | 1 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 167 | -4.17e-4 | -4.17e-4 | -4.17e-4 | +5.03e-5 |
| 144 | 3.00e-2 | 2 | 4.62e-2 | 4.69e-2 | 4.66e-2 | 4.69e-2 | 137 | +1.14e-4 | +4.22e-4 | +2.68e-4 | +9.01e-5 |
| 145 | 3.00e-2 | 2 | 4.27e-2 | 4.58e-2 | 4.42e-2 | 4.58e-2 | 130 | -5.74e-4 | +5.39e-4 | -1.73e-5 | +7.53e-5 |
| 146 | 3.00e-2 | 2 | 4.18e-2 | 4.79e-2 | 4.48e-2 | 4.79e-2 | 130 | -5.68e-4 | +1.05e-3 | +2.41e-4 | +1.15e-4 |
| 147 | 3.00e-2 | 2 | 4.39e-2 | 4.86e-2 | 4.63e-2 | 4.86e-2 | 130 | -5.31e-4 | +7.71e-4 | +1.20e-4 | +1.22e-4 |
| 148 | 3.00e-2 | 2 | 4.55e-2 | 4.86e-2 | 4.70e-2 | 4.86e-2 | 130 | -3.72e-4 | +5.20e-4 | +7.40e-5 | +1.18e-4 |
| 149 | 3.00e-2 | 2 | 4.42e-2 | 5.04e-2 | 4.73e-2 | 5.04e-2 | 130 | -5.81e-4 | +1.00e-3 | +2.10e-4 | +1.43e-4 |
| 150 | 3.00e-3 | 2 | 1.64e-2 | 4.52e-2 | 3.08e-2 | 1.64e-2 | 113 | -8.95e-3 | -6.39e-4 | -4.80e-3 | -8.37e-4 |
| 151 | 3.00e-3 | 2 | 3.89e-3 | 4.29e-3 | 4.09e-3 | 4.29e-3 | 105 | -9.74e-3 | +9.37e-4 | -4.40e-3 | -1.46e-3 |
| 152 | 3.00e-3 | 3 | 3.75e-3 | 4.25e-3 | 3.95e-3 | 3.84e-3 | 103 | -9.97e-4 | +1.08e-3 | -3.04e-4 | -1.15e-3 |
| 153 | 3.00e-3 | 2 | 3.51e-3 | 3.96e-3 | 3.74e-3 | 3.96e-3 | 103 | -6.75e-4 | +1.18e-3 | +2.52e-4 | -8.73e-4 |
| 154 | 3.00e-3 | 2 | 3.68e-3 | 4.39e-3 | 4.03e-3 | 4.39e-3 | 106 | -4.43e-4 | +1.65e-3 | +6.02e-4 | -5.82e-4 |
| 155 | 3.00e-3 | 2 | 3.60e-3 | 4.35e-3 | 3.98e-3 | 4.35e-3 | 103 | -1.26e-3 | +1.83e-3 | +2.83e-4 | -4.02e-4 |
| 156 | 3.00e-3 | 3 | 3.22e-3 | 3.83e-3 | 3.57e-3 | 3.22e-3 | 85 | -2.06e-3 | +5.64e-4 | -9.90e-4 | -5.68e-4 |
| 157 | 3.00e-3 | 3 | 3.17e-3 | 3.65e-3 | 3.39e-3 | 3.35e-3 | 106 | -8.02e-4 | +1.61e-3 | +2.19e-4 | -3.61e-4 |
| 158 | 3.00e-3 | 3 | 3.34e-3 | 4.07e-3 | 3.72e-3 | 3.34e-3 | 95 | -2.08e-3 | +9.05e-4 | -1.37e-4 | -3.28e-4 |
| 159 | 3.00e-3 | 2 | 3.64e-3 | 3.74e-3 | 3.69e-3 | 3.74e-3 | 84 | +3.31e-4 | +6.98e-4 | +5.15e-4 | -1.70e-4 |
| 160 | 3.00e-3 | 3 | 3.27e-3 | 4.05e-3 | 3.56e-3 | 3.27e-3 | 84 | -2.57e-3 | +2.28e-3 | -3.78e-4 | -2.44e-4 |
| 161 | 3.00e-3 | 3 | 3.20e-3 | 4.24e-3 | 3.58e-3 | 3.30e-3 | 79 | -3.16e-3 | +3.37e-3 | +1.44e-5 | -2.04e-4 |
| 162 | 3.00e-3 | 5 | 2.99e-3 | 4.03e-3 | 3.26e-3 | 3.10e-3 | 70 | -4.04e-3 | +3.34e-3 | -1.53e-4 | -1.95e-4 |
| 163 | 3.00e-3 | 2 | 3.08e-3 | 3.95e-3 | 3.52e-3 | 3.95e-3 | 60 | -5.30e-5 | +4.15e-3 | +2.05e-3 | +2.53e-4 |
| 164 | 3.00e-3 | 4 | 2.70e-3 | 3.59e-3 | 2.98e-3 | 2.74e-3 | 60 | -4.49e-3 | +3.59e-3 | -1.05e-3 | -1.81e-4 |
| 165 | 3.00e-3 | 4 | 2.66e-3 | 3.72e-3 | 3.13e-3 | 2.98e-3 | 68 | -2.42e-3 | +4.66e-3 | +2.60e-4 | -7.12e-5 |
| 166 | 3.00e-3 | 5 | 2.37e-3 | 3.56e-3 | 2.80e-3 | 2.37e-3 | 46 | -5.59e-3 | +2.87e-3 | -1.07e-3 | -5.26e-4 |
| 167 | 3.00e-3 | 5 | 2.28e-3 | 3.45e-3 | 2.59e-3 | 2.28e-3 | 45 | -8.48e-3 | +7.53e-3 | -2.54e-4 | -4.99e-4 |
| 168 | 3.00e-3 | 6 | 1.90e-3 | 3.28e-3 | 2.46e-3 | 1.90e-3 | 38 | -6.54e-3 | +6.23e-3 | -1.09e-3 | -9.56e-4 |
| 169 | 3.00e-3 | 8 | 1.88e-3 | 3.14e-3 | 2.13e-3 | 1.88e-3 | 30 | -9.73e-3 | +1.28e-2 | -1.25e-4 | -6.31e-4 |
| 170 | 3.00e-3 | 11 | 1.32e-3 | 2.82e-3 | 1.61e-3 | 1.34e-3 | 20 | -3.20e-2 | +2.09e-2 | -1.39e-3 | -1.23e-3 |
| 171 | 3.00e-3 | 21 | 9.34e-4 | 2.61e-3 | 1.28e-3 | 1.38e-3 | 20 | -5.07e-2 | +3.77e-2 | -3.55e-4 | +2.36e-4 |
| 172 | 3.00e-3 | 9 | 1.07e-3 | 2.72e-3 | 1.34e-3 | 1.15e-3 | 18 | -4.29e-2 | +4.01e-2 | -1.72e-3 | -8.95e-4 |
| 173 | 3.00e-3 | 12 | 1.01e-3 | 2.69e-3 | 1.43e-3 | 1.51e-3 | 20 | -6.03e-2 | +5.05e-2 | +2.75e-4 | -1.69e-4 |
| 174 | 3.00e-3 | 15 | 1.06e-3 | 2.80e-3 | 1.33e-3 | 1.21e-3 | 16 | -5.26e-2 | +3.57e-2 | -8.00e-4 | -5.18e-4 |
| 175 | 3.00e-3 | 12 | 1.01e-3 | 2.73e-3 | 1.66e-3 | 1.96e-3 | 32 | -5.88e-2 | +5.13e-2 | +4.57e-4 | +1.81e-4 |
| 176 | 3.00e-3 | 7 | 1.65e-3 | 3.01e-3 | 1.98e-3 | 1.65e-3 | 28 | -1.83e-2 | +1.70e-2 | -6.60e-4 | -4.51e-4 |
| 177 | 3.00e-3 | 13 | 1.16e-3 | 2.73e-3 | 1.53e-3 | 1.16e-3 | 17 | -1.97e-2 | +2.11e-2 | -1.65e-3 | -1.67e-3 |
| 178 | 3.00e-3 | 16 | 1.04e-3 | 2.46e-3 | 1.29e-3 | 1.23e-3 | 18 | -4.56e-2 | +4.11e-2 | -8.01e-5 | -4.01e-4 |
| 179 | 3.00e-3 | 15 | 1.03e-3 | 2.69e-3 | 1.32e-3 | 1.53e-3 | 20 | -4.73e-2 | +5.15e-2 | +8.33e-4 | +9.35e-4 |
| 180 | 3.00e-3 | 1 | 1.44e-3 | 1.44e-3 | 1.44e-3 | 1.44e-3 | 18 | -3.57e-3 | -3.57e-3 | -3.57e-3 | +4.84e-4 |
| 181 | 3.00e-3 | 1 | 1.37e-3 | 1.37e-3 | 1.37e-3 | 1.37e-3 | 306 | -1.48e-4 | -1.48e-4 | -1.48e-4 | +4.21e-4 |
| 182 | 3.00e-3 | 1 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 330 | +4.75e-3 | +4.75e-3 | +4.75e-3 | +8.54e-4 |
| 183 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 291 | +9.10e-5 | +9.10e-5 | +9.10e-5 | +7.77e-4 |
| 184 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 365 | -4.85e-5 | -4.85e-5 | -4.85e-5 | +6.95e-4 |
| 185 | 3.00e-3 | 1 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 284 | +3.36e-4 | +3.36e-4 | +3.36e-4 | +6.59e-4 |
| 186 | 3.00e-3 | 1 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 280 | -5.05e-4 | -5.05e-4 | -5.05e-4 | +5.43e-4 |
| 187 | 3.00e-3 | 1 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 261 | -6.63e-5 | -6.63e-5 | -6.63e-5 | +4.82e-4 |
| 188 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 288 | +1.23e-4 | +1.23e-4 | +1.23e-4 | +4.46e-4 |
| 189 | 3.00e-3 | 1 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 294 | -5.96e-5 | -5.96e-5 | -5.96e-5 | +3.95e-4 |
| 190 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 275 | +1.28e-4 | +1.28e-4 | +1.28e-4 | +3.68e-4 |
| 191 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 256 | -1.44e-4 | -1.44e-4 | -1.44e-4 | +3.17e-4 |
| 193 | 3.00e-3 | 2 | 6.24e-3 | 6.81e-3 | 6.52e-3 | 6.81e-3 | 252 | -5.31e-5 | +3.45e-4 | +1.46e-4 | +2.87e-4 |
| 195 | 3.00e-3 | 2 | 6.23e-3 | 6.50e-3 | 6.36e-3 | 6.50e-3 | 252 | -2.98e-4 | +1.72e-4 | -6.32e-5 | +2.23e-4 |
| 196 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 288 | -2.11e-4 | -2.11e-4 | -2.11e-4 | +1.79e-4 |
| 197 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 304 | +2.04e-4 | +2.04e-4 | +2.04e-4 | +1.82e-4 |
| 199 | 3.00e-3 | 1 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 366 | +2.06e-4 | +2.06e-4 | +2.06e-4 | +1.84e-4 |

