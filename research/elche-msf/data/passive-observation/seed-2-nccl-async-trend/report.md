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
| nccl-async | 0.049580 | 0.9206 | +0.0081 | 1999.5 | 553 | 41.8 | 100% | 100% | 6.1 |

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
| nccl-async | 1.9943 | 0.8050 | 0.6201 | 0.5583 | 0.5263 | 0.5023 | 0.4750 | 0.4587 | 0.3850 | 0.3856 | 0.1986 | 0.1588 | 0.1463 | 0.1288 | 0.1260 | 0.0677 | 0.0611 | 0.0571 | 0.0521 | 0.0496 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3990 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3044 | 3.2 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2967 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 398 | 387 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1998.4 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu2 | 1998.4 | 1.0 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 1.3s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 2.7s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 2.1s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 363 | 0 | 553 | 41.8 | 2105/9857 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 175.8 | 8.8% |

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
| resnet-graph | nccl-async | 195 | 553 | 0 | 4.33e-3 | +6.37e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 553 | 8.22e-2 | 5.78e-2 | 0.00e0 | 4.53e-1 | 37.8 | -1.67e-4 | 3.21e-3 |
| resnet-graph | nccl-async | 1 | 553 | 8.46e-2 | 6.58e-2 | 0.00e0 | 7.42e-1 | 41.4 | -2.01e-4 | 4.59e-3 |
| resnet-graph | nccl-async | 2 | 553 | 8.23e-2 | 5.97e-2 | 0.00e0 | 4.33e-1 | 20.8 | -1.62e-4 | 4.41e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9728 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9923 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9726 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 50 (0,1,2,3,4,5,7,13…142,147) | 0 (—) | — | 0,1,2,3,4,5,7,13…142,147 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 34 | 34 |
| resnet-graph | nccl-async | 0e0 | 5 | 18 | 18 |
| resnet-graph | nccl-async | 0e0 | 10 | 4 | 4 |
| resnet-graph | nccl-async | 1e-4 | 3 | 12 | 12 |
| resnet-graph | nccl-async | 1e-4 | 5 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 10 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 414 | +0.070 |
| resnet-graph | nccl-async | 3.00e-2 | 99–148 | 50 | +0.143 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 84 | -0.013 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 550 | +0.011 | 194 | +0.219 | +0.409 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 551 | 3.49e1–8.07e1 | 6.79e1 | 2.57e-3 | 6.38e-3 | 8.82e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 416 | 58–76681 | -1.139e-6 | 0.004 | -1.208e-6 | 0.005 | 97 | -5.824e-6 | 0.103 | 26–833 | +1.772e-3 | 0.605 |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | 401 | 910–76681 | -1.968e-6 | 0.014 | -2.215e-6 | 0.017 | 96 | -6.643e-6 | 0.135 | 34–833 | +1.766e-3 | 0.673 |
| resnet-graph | nccl-async | 3.00e-2 | 99–148 | 51 | 77514–115918 | +4.050e-6 | 0.010 | +3.864e-6 | 0.009 | 47 | +9.124e-6 | 0.058 | 555–959 | -9.376e-4 | 0.026 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 85 | 116620–155949 | -2.002e-5 | 0.244 | -2.034e-5 | 0.249 | 51 | -1.955e-5 | 0.214 | 233–702 | +1.586e-3 | 0.172 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | +1.772e-3 | r0: +1.739e-3, r1: +1.767e-3, r2: +1.820e-3 | r0: 0.646, r1: 0.568, r2: 0.599 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | +1.766e-3 | r0: +1.726e-3, r1: +1.771e-3, r2: +1.809e-3 | r0: 0.716, r1: 0.649, r2: 0.648 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 99–148 | -9.376e-4 | r0: -9.420e-4, r1: -9.266e-4, r2: -9.432e-4 | r0: 0.026, r1: 0.026, r2: 0.027 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | +1.586e-3 | r0: +1.597e-3, r1: +1.598e-3, r2: +1.564e-3 | r0: 0.177, r1: 0.172, r2: 0.166 | 1.02× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇██████████████████▇██▇▇▄▅▅▅▅▅▅▅▆▆▆▄▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇▇▇▇▇▇▇▇▇██▇▇▇▆█▇▇█▇▇▇▇▇▇█▇██▇▆▆▆▇▇▇▇▇██▇█▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 16 | 0.00e0 | 7.42e-1 | 1.20e-1 | 5.32e-2 | 23 | -7.29e-2 | +2.14e-2 | -1.68e-2 | -8.68e-3 |
| 1 | 3.00e-1 | 13 | 4.86e-2 | 1.06e-1 | 5.71e-2 | 5.61e-2 | 20 | -3.49e-2 | +3.38e-2 | -1.24e-4 | -1.79e-3 |
| 2 | 3.00e-1 | 13 | 5.28e-2 | 9.27e-2 | 5.92e-2 | 5.54e-2 | 17 | -2.63e-2 | +2.61e-2 | +2.16e-5 | -6.29e-4 |
| 3 | 3.00e-1 | 12 | 5.43e-2 | 1.05e-1 | 7.19e-2 | 6.47e-2 | 19 | -1.89e-2 | +2.88e-2 | +5.04e-4 | -3.76e-4 |
| 4 | 3.00e-1 | 19 | 5.21e-2 | 1.16e-1 | 6.90e-2 | 8.31e-2 | 18 | -5.42e-2 | +3.44e-2 | -1.46e-4 | +7.90e-4 |
| 5 | 3.00e-1 | 11 | 6.59e-2 | 1.29e-1 | 7.85e-2 | 6.84e-2 | 16 | -3.72e-2 | +3.20e-2 | -6.02e-4 | -4.66e-4 |
| 6 | 3.00e-1 | 2 | 6.18e-2 | 6.28e-2 | 6.23e-2 | 6.28e-2 | 190 | -6.72e-3 | +8.48e-5 | -3.32e-3 | -9.74e-4 |
| 7 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 225 | +5.04e-3 | +5.04e-3 | +5.04e-3 | -3.73e-4 |
| 8 | 3.00e-1 | 2 | 1.85e-1 | 1.89e-1 | 1.87e-1 | 1.85e-1 | 212 | -1.44e-4 | -1.05e-4 | -1.24e-4 | -3.25e-4 |
| 10 | 3.00e-1 | 2 | 1.80e-1 | 1.98e-1 | 1.89e-1 | 1.98e-1 | 229 | -8.84e-5 | +4.12e-4 | +1.62e-4 | -2.30e-4 |
| 11 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 269 | -3.58e-4 | -3.58e-4 | -3.58e-4 | -2.43e-4 |
| 12 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 263 | +2.04e-4 | +2.04e-4 | +2.04e-4 | -1.98e-4 |
| 13 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 238 | +7.43e-6 | +7.43e-6 | +7.43e-6 | -1.78e-4 |
| 14 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 243 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -1.71e-4 |
| 15 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 245 | -4.88e-5 | -4.88e-5 | -4.88e-5 | -1.59e-4 |
| 16 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 240 | -5.70e-6 | -5.70e-6 | -5.70e-6 | -1.44e-4 |
| 17 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 244 | +1.94e-5 | +1.94e-5 | +1.94e-5 | -1.28e-4 |
| 18 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 246 | +4.15e-5 | +4.15e-5 | +4.15e-5 | -1.11e-4 |
| 19 | 3.00e-1 | 2 | 1.85e-1 | 1.89e-1 | 1.87e-1 | 1.89e-1 | 217 | +1.58e-5 | +7.66e-5 | +4.62e-5 | -8.05e-5 |
| 20 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 320 | -1.96e-4 | -1.96e-4 | -1.96e-4 | -9.21e-5 |
| 21 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 238 | +5.41e-4 | +5.41e-4 | +5.41e-4 | -2.87e-5 |
| 22 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 228 | -4.13e-4 | -4.13e-4 | -4.13e-4 | -6.72e-5 |
| 23 | 3.00e-1 | 2 | 1.83e-1 | 1.84e-1 | 1.83e-1 | 1.84e-1 | 201 | -1.60e-5 | +2.09e-5 | +2.46e-6 | -5.37e-5 |
| 25 | 3.00e-1 | 2 | 1.81e-1 | 1.95e-1 | 1.88e-1 | 1.95e-1 | 185 | -5.33e-5 | +4.10e-4 | +1.79e-4 | -7.28e-6 |
| 26 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 222 | -6.09e-4 | -6.09e-4 | -6.09e-4 | -6.75e-5 |
| 27 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 238 | +2.96e-4 | +2.96e-4 | +2.96e-4 | -3.11e-5 |
| 28 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 235 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -1.45e-5 |
| 29 | 3.00e-1 | 2 | 1.85e-1 | 1.91e-1 | 1.88e-1 | 1.85e-1 | 198 | -1.39e-4 | +4.19e-5 | -4.87e-5 | -2.19e-5 |
| 30 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 240 | -2.52e-4 | -2.52e-4 | -2.52e-4 | -4.49e-5 |
| 31 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 234 | +3.91e-4 | +3.91e-4 | +3.91e-4 | -1.31e-6 |
| 32 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 243 | -2.32e-5 | -2.32e-5 | -2.32e-5 | -3.51e-6 |
| 33 | 3.00e-1 | 2 | 1.87e-1 | 1.91e-1 | 1.89e-1 | 1.87e-1 | 186 | -9.95e-5 | +1.18e-5 | -4.39e-5 | -1.17e-5 |
| 34 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 220 | -2.57e-4 | -2.57e-4 | -2.57e-4 | -3.63e-5 |
| 35 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 218 | +1.77e-4 | +1.77e-4 | +1.77e-4 | -1.50e-5 |
| 36 | 3.00e-1 | 2 | 1.84e-1 | 1.89e-1 | 1.87e-1 | 1.89e-1 | 175 | +4.08e-6 | +1.67e-4 | +8.54e-5 | +4.91e-6 |
| 37 | 3.00e-1 | 1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 220 | -4.73e-4 | -4.73e-4 | -4.73e-4 | -4.29e-5 |
| 38 | 3.00e-1 | 2 | 1.77e-1 | 1.90e-1 | 1.83e-1 | 1.77e-1 | 166 | -4.19e-4 | +5.84e-4 | +8.24e-5 | -2.41e-5 |
| 39 | 3.00e-1 | 1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 210 | -3.04e-4 | -3.04e-4 | -3.04e-4 | -5.21e-5 |
| 40 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 230 | +4.85e-4 | +4.85e-4 | +4.85e-4 | +1.64e-6 |
| 41 | 3.00e-1 | 2 | 1.91e-1 | 1.95e-1 | 1.93e-1 | 1.95e-1 | 168 | +1.03e-4 | +1.36e-4 | +1.20e-4 | +2.42e-5 |
| 42 | 3.00e-1 | 2 | 1.73e-1 | 1.82e-1 | 1.77e-1 | 1.82e-1 | 145 | -6.10e-4 | +3.42e-4 | -1.34e-4 | -1.11e-6 |
| 43 | 3.00e-1 | 1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 160 | -6.54e-4 | -6.54e-4 | -6.54e-4 | -6.64e-5 |
| 44 | 3.00e-1 | 2 | 1.72e-1 | 1.74e-1 | 1.73e-1 | 1.74e-1 | 158 | +5.98e-5 | +3.03e-4 | +1.82e-4 | -2.05e-5 |
| 45 | 3.00e-1 | 2 | 1.69e-1 | 1.83e-1 | 1.76e-1 | 1.83e-1 | 168 | -1.49e-4 | +4.74e-4 | +1.63e-4 | +1.74e-5 |
| 46 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 222 | -2.35e-4 | -2.35e-4 | -2.35e-4 | -7.90e-6 |
| 47 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 208 | +5.13e-4 | +5.13e-4 | +5.13e-4 | +4.42e-5 |
| 48 | 3.00e-1 | 2 | 1.85e-1 | 1.87e-1 | 1.86e-1 | 1.87e-1 | 168 | -2.29e-4 | +6.53e-5 | -8.20e-5 | +2.17e-5 |
| 49 | 3.00e-1 | 2 | 1.72e-1 | 1.92e-1 | 1.82e-1 | 1.92e-1 | 159 | -3.88e-4 | +6.80e-4 | +1.46e-4 | +5.06e-5 |
| 50 | 3.00e-1 | 1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 192 | -6.83e-4 | -6.83e-4 | -6.83e-4 | -2.27e-5 |
| 51 | 3.00e-1 | 2 | 1.87e-1 | 1.89e-1 | 1.88e-1 | 1.89e-1 | 159 | +7.05e-5 | +5.22e-4 | +2.96e-4 | +3.57e-5 |
| 52 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 200 | -3.94e-4 | -3.94e-4 | -3.94e-4 | -7.29e-6 |
| 53 | 3.00e-1 | 2 | 1.81e-1 | 1.88e-1 | 1.85e-1 | 1.81e-1 | 134 | -2.94e-4 | +4.07e-4 | +5.65e-5 | +1.32e-6 |
| 54 | 3.00e-1 | 2 | 1.62e-1 | 1.80e-1 | 1.71e-1 | 1.80e-1 | 129 | -6.45e-4 | +8.20e-4 | +8.74e-5 | +2.50e-5 |
| 55 | 3.00e-1 | 2 | 1.63e-1 | 1.74e-1 | 1.68e-1 | 1.74e-1 | 126 | -5.98e-4 | +4.80e-4 | -5.93e-5 | +1.44e-5 |
| 56 | 3.00e-1 | 2 | 1.62e-1 | 1.76e-1 | 1.69e-1 | 1.76e-1 | 126 | -4.05e-4 | +6.32e-4 | +1.13e-4 | +3.84e-5 |
| 57 | 3.00e-1 | 2 | 1.62e-1 | 1.73e-1 | 1.67e-1 | 1.73e-1 | 127 | -5.14e-4 | +4.96e-4 | -8.70e-6 | +3.45e-5 |
| 58 | 3.00e-1 | 1 | 1.65e-1 | 1.65e-1 | 1.65e-1 | 1.65e-1 | 175 | -2.49e-4 | -2.49e-4 | -2.49e-4 | +6.18e-6 |
| 59 | 3.00e-1 | 2 | 1.80e-1 | 1.82e-1 | 1.81e-1 | 1.80e-1 | 138 | -1.07e-4 | +5.78e-4 | +2.36e-4 | +4.64e-5 |
| 60 | 3.00e-1 | 2 | 1.69e-1 | 1.76e-1 | 1.72e-1 | 1.76e-1 | 116 | -3.98e-4 | +3.65e-4 | -1.67e-5 | +3.82e-5 |
| 61 | 3.00e-1 | 3 | 1.53e-1 | 1.83e-1 | 1.64e-1 | 1.57e-1 | 112 | -1.34e-3 | +1.58e-3 | -1.93e-4 | -3.02e-5 |
| 62 | 3.00e-1 | 2 | 1.56e-1 | 1.81e-1 | 1.69e-1 | 1.81e-1 | 124 | -6.11e-5 | +1.24e-3 | +5.89e-4 | +9.40e-5 |
| 63 | 3.00e-1 | 2 | 1.61e-1 | 1.73e-1 | 1.67e-1 | 1.73e-1 | 110 | -8.27e-4 | +6.32e-4 | -9.72e-5 | +6.49e-5 |
| 64 | 3.00e-1 | 2 | 1.56e-1 | 1.73e-1 | 1.65e-1 | 1.73e-1 | 110 | -7.08e-4 | +9.10e-4 | +1.01e-4 | +7.99e-5 |
| 65 | 3.00e-1 | 2 | 1.50e-1 | 1.69e-1 | 1.59e-1 | 1.69e-1 | 110 | -1.10e-3 | +1.12e-3 | +1.19e-5 | +7.80e-5 |
| 66 | 3.00e-1 | 3 | 1.56e-1 | 1.70e-1 | 1.61e-1 | 1.56e-1 | 117 | -7.56e-4 | +7.60e-4 | -1.98e-4 | +1.22e-6 |
| 67 | 3.00e-1 | 2 | 1.63e-1 | 1.85e-1 | 1.74e-1 | 1.85e-1 | 105 | +2.65e-4 | +1.17e-3 | +7.18e-4 | +1.42e-4 |
| 68 | 3.00e-1 | 2 | 1.53e-1 | 1.69e-1 | 1.61e-1 | 1.69e-1 | 101 | -1.37e-3 | +9.80e-4 | -1.93e-4 | +9.00e-5 |
| 69 | 3.00e-1 | 2 | 1.52e-1 | 1.72e-1 | 1.62e-1 | 1.72e-1 | 92 | -7.52e-4 | +1.33e-3 | +2.87e-4 | +1.38e-4 |
| 70 | 3.00e-1 | 4 | 1.41e-1 | 1.77e-1 | 1.52e-1 | 1.41e-1 | 90 | -2.50e-3 | +2.17e-3 | -4.00e-4 | -5.66e-5 |
| 71 | 3.00e-1 | 2 | 1.47e-1 | 1.71e-1 | 1.59e-1 | 1.71e-1 | 90 | +3.03e-4 | +1.70e-3 | +1.00e-3 | +1.52e-4 |
| 72 | 3.00e-1 | 4 | 1.30e-1 | 1.65e-1 | 1.45e-1 | 1.36e-1 | 72 | -3.26e-3 | +1.60e-3 | -5.68e-4 | -9.19e-5 |
| 73 | 3.00e-1 | 3 | 1.35e-1 | 1.61e-1 | 1.44e-1 | 1.36e-1 | 78 | -2.16e-3 | +2.35e-3 | +2.66e-5 | -8.05e-5 |
| 74 | 3.00e-1 | 3 | 1.36e-1 | 1.67e-1 | 1.47e-1 | 1.36e-1 | 72 | -2.91e-3 | +2.70e-3 | -3.81e-5 | -9.88e-5 |
| 75 | 3.00e-1 | 3 | 1.29e-1 | 1.62e-1 | 1.41e-1 | 1.32e-1 | 75 | -2.77e-3 | +3.19e-3 | -3.23e-6 | -9.67e-5 |
| 76 | 3.00e-1 | 4 | 1.22e-1 | 1.66e-1 | 1.39e-1 | 1.30e-1 | 59 | -5.22e-3 | +2.95e-3 | -2.00e-4 | -1.57e-4 |
| 77 | 3.00e-1 | 5 | 1.24e-1 | 1.58e-1 | 1.36e-1 | 1.34e-1 | 68 | -2.61e-3 | +3.93e-3 | +1.88e-4 | -3.70e-5 |
| 78 | 3.00e-1 | 3 | 1.25e-1 | 1.65e-1 | 1.41e-1 | 1.25e-1 | 64 | -4.39e-3 | +3.28e-3 | -3.64e-4 | -1.69e-4 |
| 79 | 3.00e-1 | 4 | 1.24e-1 | 1.62e-1 | 1.35e-1 | 1.24e-1 | 48 | -4.17e-3 | +4.07e-3 | -5.76e-5 | -1.75e-4 |
| 80 | 3.00e-1 | 7 | 1.04e-1 | 1.44e-1 | 1.13e-1 | 1.07e-1 | 42 | -6.92e-3 | +6.79e-3 | -1.84e-4 | -2.02e-4 |
| 81 | 3.00e-1 | 5 | 9.58e-2 | 1.49e-1 | 1.12e-1 | 9.58e-2 | 37 | -8.91e-3 | +8.62e-3 | -5.31e-4 | -4.50e-4 |
| 82 | 3.00e-1 | 10 | 7.35e-2 | 1.40e-1 | 8.92e-2 | 8.31e-2 | 22 | -1.93e-2 | +1.18e-2 | -5.66e-4 | -4.05e-4 |
| 83 | 3.00e-1 | 16 | 6.49e-2 | 1.43e-1 | 7.67e-2 | 6.75e-2 | 18 | -2.87e-2 | +2.68e-2 | -5.83e-4 | -5.90e-4 |
| 84 | 3.00e-1 | 11 | 5.97e-2 | 1.41e-1 | 7.31e-2 | 5.97e-2 | 14 | -4.03e-2 | +3.89e-2 | -9.51e-4 | -1.26e-3 |
| 85 | 3.00e-1 | 19 | 5.62e-2 | 1.37e-1 | 6.92e-2 | 7.06e-2 | 19 | -6.12e-2 | +5.78e-2 | +4.12e-4 | +1.08e-4 |
| 86 | 3.00e-1 | 11 | 5.64e-2 | 1.54e-1 | 7.56e-2 | 7.60e-2 | 17 | -5.43e-2 | +4.31e-2 | +8.21e-5 | +3.26e-4 |
| 87 | 3.00e-1 | 8 | 6.88e-2 | 1.81e-1 | 1.01e-1 | 9.26e-2 | 25 | -2.63e-2 | +3.22e-2 | +6.40e-4 | +2.60e-4 |
| 88 | 3.00e-1 | 11 | 6.95e-2 | 1.40e-1 | 9.40e-2 | 9.28e-2 | 27 | -2.90e-2 | +2.66e-2 | +3.02e-4 | +1.56e-4 |
| 89 | 3.00e-1 | 12 | 6.19e-2 | 1.40e-1 | 7.85e-2 | 8.15e-2 | 23 | -4.00e-2 | +2.70e-2 | -1.26e-6 | +3.25e-4 |
| 90 | 3.00e-1 | 14 | 5.51e-2 | 1.52e-1 | 7.29e-2 | 6.86e-2 | 17 | -3.76e-2 | +2.68e-2 | -9.31e-4 | -8.75e-5 |
| 91 | 3.00e-1 | 16 | 6.90e-2 | 1.48e-1 | 7.94e-2 | 7.59e-2 | 18 | -3.62e-2 | +4.45e-2 | +8.17e-4 | +2.08e-4 |
| 92 | 3.00e-1 | 10 | 6.39e-2 | 1.52e-1 | 7.90e-2 | 6.39e-2 | 15 | -3.51e-2 | +3.72e-2 | -1.08e-3 | -1.02e-3 |
| 93 | 3.00e-1 | 14 | 6.17e-2 | 1.46e-1 | 7.70e-2 | 7.10e-2 | 17 | -3.10e-2 | +4.09e-2 | +6.37e-4 | -1.04e-4 |
| 94 | 3.00e-1 | 17 | 5.11e-2 | 1.47e-1 | 6.74e-2 | 8.16e-2 | 17 | -7.25e-2 | +5.41e-2 | +1.38e-4 | +1.11e-3 |
| 95 | 3.00e-1 | 16 | 5.43e-2 | 1.38e-1 | 6.91e-2 | 5.63e-2 | 14 | -5.23e-2 | +4.51e-2 | -1.39e-3 | -1.39e-3 |
| 96 | 3.00e-1 | 2 | 5.60e-2 | 6.83e-2 | 6.22e-2 | 6.83e-2 | 19 | -3.49e-4 | +1.04e-2 | +5.03e-3 | -1.14e-4 |
| 97 | 3.00e-1 | 1 | 7.70e-2 | 7.70e-2 | 7.70e-2 | 7.70e-2 | 298 | +4.05e-4 | +4.05e-4 | +4.05e-4 | -6.17e-5 |
| 98 | 3.00e-1 | 1 | 2.57e-1 | 2.57e-1 | 2.57e-1 | 2.57e-1 | 290 | +4.15e-3 | +4.15e-3 | +4.15e-3 | +3.60e-4 |
| 99 | 3.00e-2 | 2 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 251 | -4.18e-4 | +3.10e-7 | -2.09e-4 | +2.54e-4 |
| 101 | 3.00e-2 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 313 | -2.22e-4 | -2.22e-4 | -2.22e-4 | +2.06e-4 |
| 102 | 3.00e-2 | 1 | 2.26e-2 | 2.26e-2 | 2.26e-2 | 2.26e-2 | 316 | -7.08e-3 | -7.08e-3 | -7.08e-3 | -5.22e-4 |
| 103 | 3.00e-2 | 1 | 2.45e-2 | 2.45e-2 | 2.45e-2 | 2.45e-2 | 279 | +2.89e-4 | +2.89e-4 | +2.89e-4 | -4.41e-4 |
| 104 | 3.00e-2 | 1 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 300 | -2.04e-5 | -2.04e-5 | -2.04e-5 | -3.99e-4 |
| 105 | 3.00e-2 | 1 | 2.63e-2 | 2.63e-2 | 2.63e-2 | 2.63e-2 | 276 | +2.80e-4 | +2.80e-4 | +2.80e-4 | -3.31e-4 |
| 106 | 3.00e-2 | 1 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 280 | -8.18e-6 | -8.18e-6 | -8.18e-6 | -2.99e-4 |
| 107 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 303 | +1.99e-4 | +1.99e-4 | +1.99e-4 | -2.49e-4 |
| 108 | 3.00e-2 | 1 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 305 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -2.08e-4 |
| 109 | 3.00e-2 | 1 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 275 | +2.28e-4 | +2.28e-4 | +2.28e-4 | -1.64e-4 |
| 110 | 3.00e-2 | 1 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 265 | -9.06e-5 | -9.06e-5 | -9.06e-5 | -1.57e-4 |
| 111 | 3.00e-2 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 338 | +4.63e-5 | +4.63e-5 | +4.63e-5 | -1.37e-4 |
| 112 | 3.00e-2 | 1 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 280 | +3.57e-4 | +3.57e-4 | +3.57e-4 | -8.74e-5 |
| 113 | 3.00e-2 | 1 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 282 | -1.51e-4 | -1.51e-4 | -1.51e-4 | -9.37e-5 |
| 114 | 3.00e-2 | 1 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 260 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -7.27e-5 |
| 115 | 3.00e-2 | 1 | 3.31e-2 | 3.31e-2 | 3.31e-2 | 3.31e-2 | 268 | -6.51e-5 | -6.51e-5 | -6.51e-5 | -7.19e-5 |
| 116 | 3.00e-2 | 1 | 3.48e-2 | 3.48e-2 | 3.48e-2 | 3.48e-2 | 256 | +1.94e-4 | +1.94e-4 | +1.94e-4 | -4.53e-5 |
| 117 | 3.00e-2 | 1 | 3.52e-2 | 3.52e-2 | 3.52e-2 | 3.52e-2 | 278 | +3.66e-5 | +3.66e-5 | +3.66e-5 | -3.71e-5 |
| 118 | 3.00e-2 | 1 | 3.57e-2 | 3.57e-2 | 3.57e-2 | 3.57e-2 | 312 | +4.89e-5 | +4.89e-5 | +4.89e-5 | -2.85e-5 |
| 119 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 338 | +2.63e-4 | +2.63e-4 | +2.63e-4 | +5.83e-7 |
| 121 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 357 | +1.45e-4 | +1.45e-4 | +1.45e-4 | +1.50e-5 |
| 122 | 3.00e-2 | 1 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 332 | +1.46e-4 | +1.46e-4 | +1.46e-4 | +2.81e-5 |
| 123 | 3.00e-2 | 1 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 279 | +6.08e-5 | +6.08e-5 | +6.08e-5 | +3.14e-5 |
| 124 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 256 | -2.63e-4 | -2.63e-4 | -2.63e-4 | +1.89e-6 |
| 125 | 3.00e-2 | 1 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 276 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -9.19e-6 |
| 126 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 276 | +1.99e-4 | +1.99e-4 | +1.99e-4 | +1.16e-5 |
| 127 | 3.00e-2 | 1 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 256 | +3.10e-5 | +3.10e-5 | +3.10e-5 | +1.36e-5 |
| 128 | 3.00e-2 | 2 | 4.19e-2 | 4.55e-2 | 4.37e-2 | 4.55e-2 | 242 | -3.90e-5 | +3.45e-4 | +1.53e-4 | +4.20e-5 |
| 130 | 3.00e-2 | 1 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 340 | -1.88e-4 | -1.88e-4 | -1.88e-4 | +1.90e-5 |
| 131 | 3.00e-2 | 1 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 269 | +5.78e-4 | +5.78e-4 | +5.78e-4 | +7.49e-5 |
| 132 | 3.00e-2 | 1 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 241 | -3.12e-4 | -3.12e-4 | -3.12e-4 | +3.62e-5 |
| 133 | 3.00e-2 | 2 | 4.42e-2 | 4.81e-2 | 4.61e-2 | 4.81e-2 | 227 | -1.71e-4 | +3.67e-4 | +9.78e-5 | +5.06e-5 |
| 134 | 3.00e-2 | 1 | 4.42e-2 | 4.42e-2 | 4.42e-2 | 4.42e-2 | 234 | -3.59e-4 | -3.59e-4 | -3.59e-4 | +9.61e-6 |
| 135 | 3.00e-2 | 1 | 4.66e-2 | 4.66e-2 | 4.66e-2 | 4.66e-2 | 281 | +1.90e-4 | +1.90e-4 | +1.90e-4 | +2.77e-5 |
| 136 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 258 | +3.94e-4 | +3.94e-4 | +3.94e-4 | +6.43e-5 |
| 137 | 3.00e-2 | 1 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 308 | -1.72e-4 | -1.72e-4 | -1.72e-4 | +4.07e-5 |
| 138 | 3.00e-2 | 1 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 250 | +3.28e-4 | +3.28e-4 | +3.28e-4 | +6.94e-5 |
| 139 | 3.00e-2 | 1 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 246 | -2.07e-4 | -2.07e-4 | -2.07e-4 | +4.18e-5 |
| 140 | 3.00e-2 | 1 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 262 | -5.91e-5 | -5.91e-5 | -5.91e-5 | +3.17e-5 |
| 141 | 3.00e-2 | 1 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 264 | +2.21e-4 | +2.21e-4 | +2.21e-4 | +5.07e-5 |
| 142 | 3.00e-2 | 1 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 259 | +9.90e-5 | +9.90e-5 | +9.90e-5 | +5.55e-5 |
| 143 | 3.00e-2 | 1 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 240 | -9.24e-6 | -9.24e-6 | -9.24e-6 | +4.90e-5 |
| 144 | 3.00e-2 | 2 | 5.22e-2 | 5.43e-2 | 5.33e-2 | 5.43e-2 | 201 | -1.32e-4 | +1.93e-4 | +3.09e-5 | +4.72e-5 |
| 145 | 3.00e-2 | 1 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 228 | -4.42e-4 | -4.42e-4 | -4.42e-4 | -1.71e-6 |
| 146 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 252 | +3.26e-4 | +3.26e-4 | +3.26e-4 | +3.10e-5 |
| 147 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 233 | +3.48e-4 | +3.48e-4 | +3.48e-4 | +6.27e-5 |
| 148 | 3.00e-2 | 1 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 254 | -1.87e-4 | -1.87e-4 | -1.87e-4 | +3.78e-5 |
| 149 | 3.00e-3 | 2 | 5.80e-2 | 5.82e-2 | 5.81e-2 | 5.80e-2 | 201 | -1.61e-5 | +2.23e-4 | +1.03e-4 | +4.90e-5 |
| 150 | 3.00e-3 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 247 | -4.00e-4 | -4.00e-4 | -4.00e-4 | +4.13e-6 |
| 151 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 219 | -1.04e-2 | -1.04e-2 | -1.04e-2 | -1.04e-3 |
| 152 | 3.00e-3 | 1 | 5.08e-3 | 5.08e-3 | 5.08e-3 | 5.08e-3 | 232 | -2.49e-4 | -2.49e-4 | -2.49e-4 | -9.58e-4 |
| 153 | 3.00e-3 | 1 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 235 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -8.50e-4 |
| 154 | 3.00e-3 | 1 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 238 | +2.92e-5 | +2.92e-5 | +2.92e-5 | -7.62e-4 |
| 155 | 3.00e-3 | 2 | 5.24e-3 | 5.42e-3 | 5.33e-3 | 5.24e-3 | 187 | -1.73e-4 | +1.17e-4 | -2.79e-5 | -6.24e-4 |
| 156 | 3.00e-3 | 1 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 217 | -5.51e-4 | -5.51e-4 | -5.51e-4 | -6.17e-4 |
| 157 | 3.00e-3 | 1 | 4.99e-3 | 4.99e-3 | 4.99e-3 | 4.99e-3 | 240 | +2.90e-4 | +2.90e-4 | +2.90e-4 | -5.26e-4 |
| 158 | 3.00e-3 | 2 | 5.04e-3 | 5.21e-3 | 5.13e-3 | 5.04e-3 | 187 | -1.78e-4 | +2.14e-4 | +1.83e-5 | -4.25e-4 |
| 159 | 3.00e-3 | 1 | 4.86e-3 | 4.86e-3 | 4.86e-3 | 4.86e-3 | 209 | -1.72e-4 | -1.72e-4 | -1.72e-4 | -3.99e-4 |
| 160 | 3.00e-3 | 1 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 208 | +1.96e-4 | +1.96e-4 | +1.96e-4 | -3.40e-4 |
| 161 | 3.00e-3 | 2 | 4.97e-3 | 5.00e-3 | 4.98e-3 | 5.00e-3 | 187 | -9.32e-5 | +3.36e-5 | -2.98e-5 | -2.80e-4 |
| 162 | 3.00e-3 | 1 | 4.96e-3 | 4.96e-3 | 4.96e-3 | 4.96e-3 | 229 | -3.37e-5 | -3.37e-5 | -3.37e-5 | -2.56e-4 |
| 163 | 3.00e-3 | 2 | 5.20e-3 | 5.33e-3 | 5.27e-3 | 5.20e-3 | 198 | -1.28e-4 | +3.19e-4 | +9.54e-5 | -1.91e-4 |
| 164 | 3.00e-3 | 1 | 5.02e-3 | 5.02e-3 | 5.02e-3 | 5.02e-3 | 250 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -1.86e-4 |
| 165 | 3.00e-3 | 1 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 223 | +3.48e-4 | +3.48e-4 | +3.48e-4 | -1.33e-4 |
| 166 | 3.00e-3 | 2 | 5.19e-3 | 5.36e-3 | 5.28e-3 | 5.36e-3 | 176 | -1.88e-4 | +1.78e-4 | -4.88e-6 | -1.07e-4 |
| 167 | 3.00e-3 | 1 | 4.76e-3 | 4.76e-3 | 4.76e-3 | 4.76e-3 | 205 | -5.83e-4 | -5.83e-4 | -5.83e-4 | -1.54e-4 |
| 168 | 3.00e-3 | 1 | 5.13e-3 | 5.13e-3 | 5.13e-3 | 5.13e-3 | 211 | +3.61e-4 | +3.61e-4 | +3.61e-4 | -1.03e-4 |
| 169 | 3.00e-3 | 1 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 198 | -3.23e-5 | -3.23e-5 | -3.23e-5 | -9.55e-5 |
| 170 | 3.00e-3 | 2 | 4.93e-3 | 5.15e-3 | 5.04e-3 | 5.15e-3 | 164 | -1.57e-4 | +2.58e-4 | +5.04e-5 | -6.57e-5 |
| 171 | 3.00e-3 | 1 | 4.52e-3 | 4.52e-3 | 4.52e-3 | 4.52e-3 | 223 | -5.77e-4 | -5.77e-4 | -5.77e-4 | -1.17e-4 |
| 172 | 3.00e-3 | 2 | 5.20e-3 | 5.28e-3 | 5.24e-3 | 5.20e-3 | 170 | -9.66e-5 | +8.25e-4 | +3.64e-4 | -3.00e-5 |
| 173 | 3.00e-3 | 2 | 4.70e-3 | 4.97e-3 | 4.83e-3 | 4.97e-3 | 164 | -4.98e-4 | +3.45e-4 | -7.64e-5 | -3.46e-5 |
| 174 | 3.00e-3 | 1 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 175 | -5.13e-4 | -5.13e-4 | -5.13e-4 | -8.25e-5 |
| 175 | 3.00e-3 | 2 | 4.87e-3 | 5.04e-3 | 4.95e-3 | 5.04e-3 | 152 | +2.26e-4 | +3.52e-4 | +2.89e-4 | -1.25e-5 |
| 176 | 3.00e-3 | 2 | 4.66e-3 | 4.81e-3 | 4.73e-3 | 4.66e-3 | 152 | -2.62e-4 | -2.01e-4 | -2.32e-4 | -5.38e-5 |
| 177 | 3.00e-3 | 1 | 4.48e-3 | 4.48e-3 | 4.48e-3 | 4.48e-3 | 216 | -1.84e-4 | -1.84e-4 | -1.84e-4 | -6.68e-5 |
| 178 | 3.00e-3 | 2 | 5.10e-3 | 5.34e-3 | 5.22e-3 | 5.10e-3 | 142 | -3.13e-4 | +9.10e-4 | +2.99e-4 | -3.48e-6 |
| 179 | 3.00e-3 | 1 | 4.77e-3 | 4.77e-3 | 4.77e-3 | 4.77e-3 | 166 | -4.02e-4 | -4.02e-4 | -4.02e-4 | -4.34e-5 |
| 180 | 3.00e-3 | 2 | 4.93e-3 | 5.23e-3 | 5.08e-3 | 5.23e-3 | 144 | +1.70e-4 | +4.08e-4 | +2.89e-4 | +2.10e-5 |
| 181 | 3.00e-3 | 2 | 4.43e-3 | 5.40e-3 | 4.92e-3 | 5.40e-3 | 134 | -7.64e-4 | +1.47e-3 | +3.55e-4 | +9.56e-5 |
| 182 | 3.00e-3 | 1 | 4.25e-3 | 4.25e-3 | 4.25e-3 | 4.25e-3 | 190 | -1.26e-3 | -1.26e-3 | -1.26e-3 | -4.03e-5 |
| 183 | 3.00e-3 | 2 | 5.02e-3 | 5.21e-3 | 5.11e-3 | 5.21e-3 | 148 | +2.53e-4 | +8.93e-4 | +5.73e-4 | +7.30e-5 |
| 184 | 3.00e-3 | 2 | 4.43e-3 | 4.84e-3 | 4.64e-3 | 4.84e-3 | 118 | -9.60e-4 | +7.50e-4 | -1.05e-4 | +4.78e-5 |
| 185 | 3.00e-3 | 2 | 4.26e-3 | 4.38e-3 | 4.32e-3 | 4.38e-3 | 121 | -9.07e-4 | +2.16e-4 | -3.45e-4 | -2.13e-5 |
| 186 | 3.00e-3 | 2 | 4.17e-3 | 4.65e-3 | 4.41e-3 | 4.65e-3 | 135 | -3.11e-4 | +8.15e-4 | +2.52e-4 | +3.63e-5 |
| 187 | 3.00e-3 | 2 | 4.58e-3 | 5.20e-3 | 4.89e-3 | 5.20e-3 | 131 | -8.60e-5 | +9.69e-4 | +4.42e-4 | +1.19e-4 |
| 188 | 3.00e-3 | 2 | 4.55e-3 | 4.72e-3 | 4.64e-3 | 4.72e-3 | 115 | -8.41e-4 | +3.14e-4 | -2.64e-4 | +5.17e-5 |
| 189 | 3.00e-3 | 3 | 3.90e-3 | 4.73e-3 | 4.19e-3 | 3.90e-3 | 115 | -1.68e-3 | +1.56e-3 | -4.49e-4 | -8.93e-5 |
| 190 | 3.00e-3 | 1 | 4.11e-3 | 4.11e-3 | 4.11e-3 | 4.11e-3 | 142 | +3.80e-4 | +3.80e-4 | +3.80e-4 | -4.24e-5 |
| 191 | 3.00e-3 | 2 | 4.50e-3 | 4.83e-3 | 4.67e-3 | 4.83e-3 | 124 | +5.54e-4 | +5.70e-4 | +5.62e-4 | +7.25e-5 |
| 192 | 3.00e-3 | 2 | 4.27e-3 | 4.88e-3 | 4.58e-3 | 4.88e-3 | 128 | -7.42e-4 | +1.04e-3 | +1.51e-4 | +9.63e-5 |
| 193 | 3.00e-3 | 2 | 4.28e-3 | 5.25e-3 | 4.77e-3 | 5.25e-3 | 130 | -7.13e-4 | +1.58e-3 | +4.31e-4 | +1.71e-4 |
| 194 | 3.00e-3 | 3 | 4.27e-3 | 4.90e-3 | 4.51e-3 | 4.35e-3 | 109 | -1.29e-3 | +1.02e-3 | -4.51e-4 | +4.02e-6 |
| 195 | 3.00e-3 | 2 | 4.00e-3 | 4.38e-3 | 4.19e-3 | 4.38e-3 | 111 | -5.66e-4 | +8.01e-4 | +1.17e-4 | +3.24e-5 |
| 196 | 3.00e-3 | 2 | 4.08e-3 | 4.35e-3 | 4.22e-3 | 4.35e-3 | 89 | -5.83e-4 | +7.21e-4 | +6.89e-5 | +4.58e-5 |
| 197 | 3.00e-3 | 3 | 3.44e-3 | 4.44e-3 | 3.91e-3 | 3.44e-3 | 89 | -2.87e-3 | +1.53e-3 | -7.24e-4 | -1.83e-4 |
| 198 | 3.00e-3 | 3 | 3.68e-3 | 4.29e-3 | 3.90e-3 | 3.68e-3 | 95 | -1.63e-3 | +1.47e-3 | +1.77e-4 | -1.08e-4 |
| 199 | 3.00e-3 | 2 | 3.78e-3 | 4.33e-3 | 4.06e-3 | 4.33e-3 | 103 | +2.01e-4 | +1.33e-3 | +7.68e-4 | +6.37e-5 |

