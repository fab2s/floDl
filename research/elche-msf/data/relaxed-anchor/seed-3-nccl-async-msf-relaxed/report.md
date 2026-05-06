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
| nccl-async | 0.041729 | 0.9221 | +0.0096 | 1957.9 | 691 | 39.2 | 100% | 100% | 6.9 |

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
| nccl-async | 1.9700 | 0.7128 | 0.5511 | 0.5001 | 0.4708 | 0.4493 | 0.4323 | 0.4307 | 0.4148 | 0.4202 | 0.1722 | 0.1431 | 0.1291 | 0.1222 | 0.1203 | 0.0571 | 0.0534 | 0.0486 | 0.0443 | 0.0417 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4022 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3047 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2931 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 400 | 394 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1956.7 | 1.3 | epoch-boundary(199) |
| nccl-async | gpu1 | 1956.9 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1956.7 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.8s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 2.9s |
| resnet-graph | nccl-async | gpu2 | 1.3s | 0.0s | 0.0s | 0.0s | 2.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 24 | 0 | 691 | 39.2 | 1480/11282 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 199.5 | 10.2% |

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
| resnet-graph | nccl-async | 187 | 691 | 0 | 6.88e-3 | +1.66e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 691 | 9.89e-2 | 3.45e-2 | 0.00e0 | 3.53e-1 | 44.6 | -6.32e-5 | 1.26e-3 |
| resnet-graph | nccl-async | 1 | 691 | 1.01e-1 | 3.87e-2 | 0.00e0 | 4.11e-1 | 37.8 | -7.02e-5 | 1.99e-3 |
| resnet-graph | nccl-async | 2 | 691 | 9.97e-2 | 3.82e-2 | 0.00e0 | 3.97e-1 | 17.7 | -7.08e-5 | 1.98e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9710 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9700 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9939 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 86 (0,1,2,3,4,6,8,9…148,149) | 0 (—) | — | 0,1,2,3,4,6,8,9…148,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 22 | 22 |
| resnet-graph | nccl-async | 0e0 | 5 | 7 | 7 |
| resnet-graph | nccl-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 3 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 585 | +0.032 |
| resnet-graph | nccl-async | 3.00e-2 | 99–149 | 58 | +0.134 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 43 | +0.012 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 688 | +0.018 | 186 | +0.170 | +0.332 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 689 | 3.37e1–7.91e1 | 7.10e1 | 2.07e-3 | 3.68e-3 | 5.89e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 587 | 63–77278 | +1.788e-6 | 0.056 | +1.803e-6 | 0.064 | 99 | +1.591e-6 | 0.352 | 33–253 | +1.536e-4 | 0.001 |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | 575 | 941–77278 | +1.516e-6 | 0.055 | +1.510e-6 | 0.064 | 98 | +1.426e-6 | 0.342 | 74–253 | -7.862e-6 | 0.000 |
| resnet-graph | nccl-async | 3.00e-2 | 99–149 | 59 | 77508–116534 | +2.336e-5 | 0.172 | +2.373e-5 | 0.175 | 44 | +2.860e-5 | 0.473 | 110–1073 | +6.254e-4 | 0.093 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 44 | 117428–155882 | -1.350e-5 | 0.097 | -1.351e-5 | 0.098 | 44 | -1.351e-5 | 0.098 | 787–1037 | +4.867e-4 | 0.003 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | +1.536e-4 | r0: +3.981e-4, r1: +8.525e-5, r2: -7.807e-6 | r0: 0.013, r1: 0.000, r2: 0.000 | 51.00× | ⚠ framing breaking |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | -7.862e-6 | r0: +2.413e-4, r1: -9.766e-5, r2: -1.530e-4 | r0: 0.009, r1: 0.001, r2: 0.001 | 2.47× | ⚠ framing breaking |
| resnet-graph | nccl-async | 3.00e-2 | 99–149 | +6.254e-4 | r0: +6.206e-4, r1: +6.339e-4, r2: +6.242e-4 | r0: 0.093, r1: 0.093, r2: 0.092 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +4.867e-4 | r0: +5.106e-4, r1: +4.827e-4, r2: +4.673e-4 | r0: 0.003, r1: 0.003, r2: 0.003 | 1.09× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█████████████████████████▃▅▅▅▅▅▆▆▆▆▆▂▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇█▇█▇▇▇▇▇▇▇▇▇▇▇█▇▇▇▇▇█▃▆▇▇███████▆▆▇▇▇▇█████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 4.11e-1 | 1.07e-1 | 7.27e-2 | 29 | -3.44e-2 | +5.75e-3 | -8.42e-3 | -4.92e-3 |
| 1 | 3.00e-1 | 10 | 6.96e-2 | 1.13e-1 | 8.39e-2 | 9.02e-2 | 35 | -1.67e-2 | +1.35e-2 | +5.49e-4 | -9.09e-4 |
| 2 | 3.00e-1 | 5 | 9.38e-2 | 1.17e-1 | 9.95e-2 | 9.38e-2 | 33 | -5.80e-3 | +4.62e-3 | -3.45e-5 | -5.91e-4 |
| 3 | 3.00e-1 | 8 | 9.40e-2 | 1.32e-1 | 1.02e-1 | 1.02e-1 | 33 | -9.47e-3 | +9.56e-3 | +3.08e-4 | -9.04e-5 |
| 4 | 3.00e-1 | 7 | 1.02e-1 | 1.42e-1 | 1.10e-1 | 1.06e-1 | 35 | -8.91e-3 | +9.13e-3 | +9.77e-5 | -4.74e-5 |
| 5 | 3.00e-1 | 9 | 1.02e-1 | 1.43e-1 | 1.12e-1 | 1.05e-1 | 39 | -6.39e-3 | +8.85e-3 | +4.74e-5 | -1.06e-4 |
| 6 | 3.00e-1 | 5 | 1.08e-1 | 1.40e-1 | 1.18e-1 | 1.15e-1 | 41 | -6.30e-3 | +5.42e-3 | +3.39e-4 | +2.91e-5 |
| 7 | 3.00e-1 | 6 | 9.77e-2 | 1.54e-1 | 1.13e-1 | 1.01e-1 | 35 | -9.28e-3 | +8.89e-3 | -4.78e-4 | -2.69e-4 |
| 8 | 3.00e-1 | 10 | 9.28e-2 | 1.46e-1 | 1.07e-1 | 1.01e-1 | 39 | -6.76e-3 | +8.99e-3 | +3.29e-5 | -1.33e-4 |
| 9 | 3.00e-1 | 3 | 9.42e-2 | 1.56e-1 | 1.19e-1 | 9.42e-2 | 39 | -1.30e-2 | +1.18e-2 | -2.39e-4 | -2.96e-4 |
| 10 | 3.00e-1 | 7 | 9.48e-2 | 1.37e-1 | 1.09e-1 | 1.08e-1 | 40 | -8.44e-3 | +7.00e-3 | +2.05e-4 | -7.26e-5 |
| 11 | 3.00e-1 | 6 | 1.04e-1 | 1.45e-1 | 1.15e-1 | 1.15e-1 | 44 | -6.83e-3 | +7.41e-3 | +1.78e-4 | +2.48e-5 |
| 12 | 3.00e-1 | 6 | 9.53e-2 | 1.49e-1 | 1.09e-1 | 9.87e-2 | 40 | -1.10e-2 | +8.35e-3 | -6.61e-4 | -3.31e-4 |
| 13 | 3.00e-1 | 6 | 1.03e-1 | 1.40e-1 | 1.13e-1 | 1.12e-1 | 46 | -6.56e-3 | +7.18e-3 | +3.92e-4 | -4.20e-5 |
| 14 | 3.00e-1 | 8 | 1.04e-1 | 1.46e-1 | 1.13e-1 | 1.04e-1 | 41 | -7.32e-3 | +5.50e-3 | -2.59e-4 | -2.16e-4 |
| 15 | 3.00e-1 | 4 | 1.03e-1 | 1.41e-1 | 1.14e-1 | 1.03e-1 | 39 | -7.89e-3 | +6.58e-3 | -1.90e-4 | -2.79e-4 |
| 16 | 3.00e-1 | 6 | 1.02e-1 | 1.39e-1 | 1.11e-1 | 1.10e-1 | 41 | -6.79e-3 | +7.90e-3 | +2.41e-4 | -6.76e-5 |
| 17 | 3.00e-1 | 6 | 1.03e-1 | 1.46e-1 | 1.15e-1 | 1.10e-1 | 37 | -5.87e-3 | +6.86e-3 | +5.60e-5 | -4.44e-5 |
| 18 | 3.00e-1 | 6 | 9.77e-2 | 1.43e-1 | 1.11e-1 | 1.14e-1 | 36 | -7.14e-3 | +7.94e-3 | +2.73e-4 | +1.29e-4 |
| 19 | 3.00e-1 | 6 | 9.55e-2 | 1.48e-1 | 1.11e-1 | 1.08e-1 | 41 | -1.03e-2 | +1.12e-2 | +2.71e-4 | +1.51e-4 |
| 20 | 3.00e-1 | 8 | 1.01e-1 | 1.49e-1 | 1.13e-1 | 1.09e-1 | 43 | -8.83e-3 | +8.48e-3 | +1.42e-4 | +8.38e-5 |
| 21 | 3.00e-1 | 4 | 9.56e-2 | 1.47e-1 | 1.13e-1 | 1.02e-1 | 40 | -1.19e-2 | +8.39e-3 | -5.41e-4 | -1.98e-4 |
| 22 | 3.00e-1 | 6 | 9.81e-2 | 1.50e-1 | 1.12e-1 | 1.09e-1 | 46 | -1.06e-2 | +8.66e-3 | +1.62e-4 | -7.10e-5 |
| 23 | 3.00e-1 | 5 | 1.09e-1 | 1.57e-1 | 1.22e-1 | 1.09e-1 | 44 | -6.47e-3 | +7.21e-3 | -2.85e-5 | -1.41e-4 |
| 24 | 3.00e-1 | 6 | 1.03e-1 | 1.42e-1 | 1.16e-1 | 1.08e-1 | 44 | -6.71e-3 | +6.10e-3 | +8.29e-5 | -1.03e-4 |
| 25 | 3.00e-1 | 8 | 1.04e-1 | 1.54e-1 | 1.15e-1 | 1.09e-1 | 40 | -8.36e-3 | +7.98e-3 | +8.57e-5 | -5.18e-5 |
| 26 | 3.00e-1 | 4 | 1.07e-1 | 1.44e-1 | 1.18e-1 | 1.12e-1 | 43 | -6.36e-3 | +6.69e-3 | +2.98e-4 | +2.72e-5 |
| 27 | 3.00e-1 | 5 | 1.01e-1 | 1.48e-1 | 1.17e-1 | 1.21e-1 | 46 | -8.75e-3 | +6.95e-3 | +3.45e-4 | +1.65e-4 |
| 28 | 3.00e-1 | 6 | 1.10e-1 | 1.54e-1 | 1.19e-1 | 1.10e-1 | 46 | -7.01e-3 | +6.91e-3 | -1.92e-4 | -5.39e-5 |
| 29 | 3.00e-1 | 6 | 9.65e-2 | 1.54e-1 | 1.12e-1 | 1.00e-1 | 39 | -1.02e-2 | +7.74e-3 | -5.80e-4 | -3.50e-4 |
| 30 | 3.00e-1 | 6 | 9.87e-2 | 1.52e-1 | 1.11e-1 | 1.09e-1 | 45 | -1.10e-2 | +9.86e-3 | +2.63e-4 | -1.01e-4 |
| 31 | 3.00e-1 | 6 | 1.04e-1 | 1.50e-1 | 1.17e-1 | 1.09e-1 | 44 | -8.24e-3 | +7.56e-3 | +9.08e-6 | -1.15e-4 |
| 32 | 3.00e-1 | 5 | 1.06e-1 | 1.55e-1 | 1.19e-1 | 1.06e-1 | 40 | -7.54e-3 | +7.85e-3 | -3.27e-6 | -1.65e-4 |
| 33 | 3.00e-1 | 8 | 1.02e-1 | 1.57e-1 | 1.13e-1 | 1.05e-1 | 45 | -9.90e-3 | +1.00e-2 | +2.89e-5 | -1.35e-4 |
| 34 | 3.00e-1 | 4 | 1.15e-1 | 1.52e-1 | 1.25e-1 | 1.18e-1 | 51 | -5.23e-3 | +5.81e-3 | +4.90e-4 | +2.08e-5 |
| 35 | 3.00e-1 | 7 | 1.09e-1 | 1.54e-1 | 1.18e-1 | 1.09e-1 | 40 | -6.56e-3 | +6.14e-3 | -3.06e-4 | -1.86e-4 |
| 36 | 3.00e-1 | 4 | 1.02e-1 | 1.45e-1 | 1.14e-1 | 1.07e-1 | 44 | -8.72e-3 | +8.55e-3 | +2.51e-5 | -1.67e-4 |
| 37 | 3.00e-1 | 8 | 1.03e-1 | 1.62e-1 | 1.18e-1 | 1.08e-1 | 41 | -9.82e-3 | +8.55e-3 | +1.32e-6 | -1.61e-4 |
| 38 | 3.00e-1 | 4 | 1.05e-1 | 1.56e-1 | 1.18e-1 | 1.05e-1 | 39 | -1.04e-2 | +9.23e-3 | -3.81e-4 | -3.20e-4 |
| 39 | 3.00e-1 | 5 | 1.01e-1 | 1.50e-1 | 1.15e-1 | 1.16e-1 | 45 | -9.84e-3 | +8.97e-3 | +4.80e-4 | -1.13e-5 |
| 40 | 3.00e-1 | 6 | 1.06e-1 | 1.56e-1 | 1.17e-1 | 1.06e-1 | 41 | -7.96e-3 | +7.64e-3 | -3.31e-4 | -2.22e-4 |
| 41 | 3.00e-1 | 7 | 1.05e-1 | 1.48e-1 | 1.17e-1 | 1.05e-1 | 43 | -7.06e-3 | +6.19e-3 | -1.48e-4 | -2.86e-4 |
| 42 | 3.00e-1 | 4 | 1.09e-1 | 1.54e-1 | 1.22e-1 | 1.12e-1 | 43 | -8.06e-3 | +7.24e-3 | +1.36e-4 | -2.11e-4 |
| 43 | 3.00e-1 | 6 | 1.09e-1 | 1.48e-1 | 1.20e-1 | 1.21e-1 | 45 | -5.47e-3 | +6.56e-3 | +3.39e-4 | +2.86e-5 |
| 44 | 3.00e-1 | 5 | 1.03e-1 | 1.48e-1 | 1.16e-1 | 1.03e-1 | 40 | -8.68e-3 | +5.53e-3 | -7.21e-4 | -3.31e-4 |
| 45 | 3.00e-1 | 9 | 1.02e-1 | 1.51e-1 | 1.14e-1 | 1.22e-1 | 44 | -8.43e-3 | +8.55e-3 | +3.93e-4 | +1.50e-4 |
| 46 | 3.00e-1 | 4 | 9.97e-2 | 1.49e-1 | 1.15e-1 | 1.03e-1 | 37 | -1.02e-2 | +8.18e-3 | -7.17e-4 | -1.98e-4 |
| 47 | 3.00e-1 | 5 | 1.02e-1 | 1.52e-1 | 1.16e-1 | 1.14e-1 | 44 | -8.99e-3 | +9.87e-3 | +5.06e-4 | +4.44e-5 |
| 48 | 3.00e-1 | 6 | 1.04e-1 | 1.51e-1 | 1.19e-1 | 1.15e-1 | 44 | -6.94e-3 | +6.30e-3 | +4.08e-5 | +2.44e-5 |
| 49 | 3.00e-1 | 6 | 1.08e-1 | 1.50e-1 | 1.18e-1 | 1.08e-1 | 41 | -6.44e-3 | +6.96e-3 | -1.87e-4 | -1.43e-4 |
| 50 | 3.00e-1 | 8 | 1.01e-1 | 1.53e-1 | 1.13e-1 | 1.03e-1 | 40 | -8.69e-3 | +6.92e-3 | -3.94e-4 | -3.44e-4 |
| 51 | 3.00e-1 | 4 | 1.06e-1 | 1.50e-1 | 1.18e-1 | 1.06e-1 | 40 | -8.72e-3 | +7.51e-3 | -3.23e-5 | -3.21e-4 |
| 52 | 3.00e-1 | 6 | 1.06e-1 | 1.50e-1 | 1.17e-1 | 1.09e-1 | 44 | -7.55e-3 | +8.20e-3 | +1.43e-4 | -1.88e-4 |
| 53 | 3.00e-1 | 6 | 1.02e-1 | 1.45e-1 | 1.16e-1 | 1.12e-1 | 47 | -8.82e-3 | +5.97e-3 | -6.51e-6 | -1.32e-4 |
| 54 | 3.00e-1 | 6 | 1.14e-1 | 1.56e-1 | 1.25e-1 | 1.20e-1 | 47 | -4.52e-3 | +5.95e-3 | +1.25e-4 | -6.53e-5 |
| 55 | 3.00e-1 | 5 | 1.08e-1 | 1.61e-1 | 1.20e-1 | 1.08e-1 | 45 | -9.22e-3 | +7.20e-3 | -4.19e-4 | -2.62e-4 |
| 56 | 3.00e-1 | 5 | 1.08e-1 | 1.51e-1 | 1.19e-1 | 1.08e-1 | 41 | -8.17e-3 | +5.14e-3 | -3.72e-4 | -3.67e-4 |
| 57 | 3.00e-1 | 6 | 1.03e-1 | 1.55e-1 | 1.15e-1 | 1.03e-1 | 42 | -8.90e-3 | +8.29e-3 | -2.54e-4 | -4.03e-4 |
| 58 | 3.00e-1 | 6 | 1.01e-1 | 1.56e-1 | 1.16e-1 | 1.10e-1 | 39 | -8.73e-3 | +8.64e-3 | +2.26e-4 | -1.68e-4 |
| 59 | 3.00e-1 | 8 | 1.02e-1 | 1.46e-1 | 1.16e-1 | 1.08e-1 | 41 | -8.45e-3 | +8.82e-3 | +9.20e-5 | -1.26e-4 |
| 60 | 3.00e-1 | 4 | 1.07e-1 | 1.50e-1 | 1.19e-1 | 1.10e-1 | 50 | -7.08e-3 | +7.19e-3 | +1.72e-4 | -8.26e-5 |
| 61 | 3.00e-1 | 5 | 1.14e-1 | 1.56e-1 | 1.27e-1 | 1.23e-1 | 53 | -5.95e-3 | +4.31e-3 | +1.78e-4 | -1.13e-5 |
| 62 | 3.00e-1 | 8 | 1.08e-1 | 1.65e-1 | 1.21e-1 | 1.10e-1 | 43 | -7.05e-3 | +5.77e-3 | -3.73e-4 | -2.54e-4 |
| 63 | 3.00e-1 | 3 | 1.08e-1 | 1.54e-1 | 1.24e-1 | 1.10e-1 | 45 | -7.52e-3 | +7.89e-3 | +6.86e-5 | -2.40e-4 |
| 64 | 3.00e-1 | 6 | 1.10e-1 | 1.51e-1 | 1.19e-1 | 1.15e-1 | 42 | -7.49e-3 | +6.91e-3 | +8.59e-5 | -1.25e-4 |
| 65 | 3.00e-1 | 5 | 1.02e-1 | 1.54e-1 | 1.19e-1 | 1.19e-1 | 48 | -9.08e-3 | +7.64e-3 | +2.70e-4 | +2.20e-5 |
| 66 | 3.00e-1 | 5 | 1.14e-1 | 1.63e-1 | 1.28e-1 | 1.22e-1 | 48 | -6.89e-3 | +5.96e-3 | +8.95e-5 | +1.33e-5 |
| 67 | 3.00e-1 | 5 | 1.02e-1 | 1.60e-1 | 1.25e-1 | 1.02e-1 | 43 | -5.76e-3 | +6.05e-3 | -6.21e-4 | -3.67e-4 |
| 68 | 3.00e-1 | 6 | 1.03e-1 | 1.61e-1 | 1.16e-1 | 1.08e-1 | 40 | -1.15e-2 | +9.12e-3 | -1.52e-5 | -2.60e-4 |
| 69 | 3.00e-1 | 6 | 1.06e-1 | 1.55e-1 | 1.19e-1 | 1.21e-1 | 43 | -8.16e-3 | +8.24e-3 | +3.88e-4 | +2.56e-5 |
| 70 | 3.00e-1 | 6 | 9.75e-2 | 1.53e-1 | 1.15e-1 | 1.11e-1 | 36 | -9.44e-3 | +7.49e-3 | -4.20e-4 | -1.69e-4 |
| 71 | 3.00e-1 | 6 | 1.00e-1 | 1.55e-1 | 1.14e-1 | 1.10e-1 | 41 | -1.01e-2 | +1.07e-2 | +1.58e-4 | -5.26e-5 |
| 72 | 3.00e-1 | 7 | 1.04e-1 | 1.50e-1 | 1.16e-1 | 1.16e-1 | 46 | -8.83e-3 | +8.34e-3 | +1.85e-4 | +4.89e-5 |
| 73 | 3.00e-1 | 4 | 1.11e-1 | 1.59e-1 | 1.25e-1 | 1.13e-1 | 46 | -8.09e-3 | +7.13e-3 | -1.16e-4 | -6.88e-5 |
| 74 | 3.00e-1 | 6 | 9.97e-2 | 1.60e-1 | 1.19e-1 | 1.09e-1 | 46 | -1.11e-2 | +7.92e-3 | -1.06e-4 | -1.48e-4 |
| 75 | 3.00e-1 | 5 | 1.11e-1 | 1.48e-1 | 1.24e-1 | 1.21e-1 | 54 | -5.78e-3 | +5.59e-3 | +3.83e-4 | +2.60e-5 |
| 76 | 3.00e-1 | 6 | 1.04e-1 | 1.60e-1 | 1.21e-1 | 1.04e-1 | 40 | -9.69e-3 | +5.74e-3 | -6.67e-4 | -3.70e-4 |
| 77 | 3.00e-1 | 7 | 1.02e-1 | 1.51e-1 | 1.15e-1 | 1.12e-1 | 46 | -9.72e-3 | +7.81e-3 | +8.19e-5 | -1.77e-4 |
| 78 | 3.00e-1 | 5 | 1.05e-1 | 1.54e-1 | 1.20e-1 | 1.15e-1 | 43 | -8.65e-3 | +7.29e-3 | +2.47e-4 | -4.68e-5 |
| 79 | 3.00e-1 | 5 | 1.11e-1 | 1.67e-1 | 1.25e-1 | 1.16e-1 | 51 | -8.20e-3 | +8.68e-3 | -1.90e-5 | -9.83e-5 |
| 80 | 3.00e-1 | 6 | 1.02e-1 | 1.52e-1 | 1.18e-1 | 1.15e-1 | 40 | -8.34e-3 | +5.39e-3 | +2.12e-5 | -3.40e-5 |
| 81 | 3.00e-1 | 8 | 1.02e-1 | 1.62e-1 | 1.13e-1 | 1.05e-1 | 48 | -9.17e-3 | +1.05e-2 | -9.66e-5 | -1.25e-4 |
| 82 | 3.00e-1 | 4 | 9.88e-2 | 1.57e-1 | 1.22e-1 | 9.88e-2 | 40 | -9.28e-3 | +5.93e-3 | -1.21e-3 | -6.28e-4 |
| 83 | 3.00e-1 | 5 | 1.08e-1 | 1.60e-1 | 1.20e-1 | 1.11e-1 | 44 | -9.18e-3 | +8.88e-3 | +1.07e-4 | -4.16e-4 |
| 84 | 3.00e-1 | 8 | 1.04e-1 | 1.57e-1 | 1.19e-1 | 1.17e-1 | 40 | -1.06e-2 | +9.21e-3 | +1.10e-4 | -1.57e-4 |
| 85 | 3.00e-1 | 4 | 1.04e-1 | 1.59e-1 | 1.20e-1 | 1.14e-1 | 40 | -1.06e-2 | +9.61e-3 | +2.44e-5 | -1.38e-4 |
| 86 | 3.00e-1 | 6 | 1.07e-1 | 1.60e-1 | 1.17e-1 | 1.07e-1 | 42 | -9.26e-3 | +9.63e-3 | -1.30e-5 | -1.48e-4 |
| 87 | 3.00e-1 | 6 | 1.02e-1 | 1.70e-1 | 1.19e-1 | 1.14e-1 | 46 | -1.22e-2 | +8.95e-3 | -6.10e-5 | -1.31e-4 |
| 88 | 3.00e-1 | 5 | 1.12e-1 | 1.70e-1 | 1.25e-1 | 1.17e-1 | 46 | -8.80e-3 | +8.21e-3 | +3.13e-5 | -1.12e-4 |
| 89 | 3.00e-1 | 8 | 1.08e-1 | 1.52e-1 | 1.19e-1 | 1.21e-1 | 48 | -7.43e-3 | +6.31e-3 | +1.31e-4 | +3.56e-5 |
| 90 | 3.00e-1 | 3 | 9.55e-2 | 1.68e-1 | 1.28e-1 | 9.55e-2 | 35 | -1.61e-2 | +9.07e-3 | -2.40e-3 | -7.81e-4 |
| 91 | 3.00e-1 | 6 | 9.56e-2 | 1.43e-1 | 1.11e-1 | 1.14e-1 | 43 | -9.89e-3 | +1.09e-2 | +7.49e-4 | -1.09e-4 |
| 92 | 3.00e-1 | 5 | 1.06e-1 | 1.58e-1 | 1.22e-1 | 1.19e-1 | 50 | -8.71e-3 | +8.59e-3 | +3.71e-4 | +5.04e-5 |
| 93 | 3.00e-1 | 9 | 1.03e-1 | 1.65e-1 | 1.18e-1 | 1.15e-1 | 48 | -9.19e-3 | +5.72e-3 | -2.91e-4 | -9.90e-5 |
| 94 | 3.00e-1 | 3 | 1.07e-1 | 1.64e-1 | 1.29e-1 | 1.07e-1 | 42 | -1.01e-2 | +8.61e-3 | -4.95e-4 | -3.07e-4 |
| 95 | 3.00e-1 | 6 | 1.05e-1 | 1.48e-1 | 1.16e-1 | 1.11e-1 | 43 | -7.88e-3 | +7.09e-3 | +6.30e-5 | -1.79e-4 |
| 96 | 3.00e-1 | 5 | 1.08e-1 | 1.72e-1 | 1.26e-1 | 1.22e-1 | 44 | -9.82e-3 | +9.74e-3 | +5.31e-4 | +6.59e-5 |
| 97 | 3.00e-1 | 7 | 1.12e-1 | 1.52e-1 | 1.20e-1 | 1.18e-1 | 48 | -6.04e-3 | +7.01e-3 | +1.89e-5 | +2.60e-5 |
| 98 | 3.00e-1 | 4 | 1.02e-1 | 1.60e-1 | 1.25e-1 | 1.20e-1 | 42 | -9.86e-3 | +6.91e-3 | +2.65e-4 | +9.07e-5 |
| 99 | 3.00e-2 | 7 | 1.10e-1 | 1.63e-1 | 1.20e-1 | 1.11e-1 | 40 | -8.73e-3 | +8.32e-3 | -2.53e-4 | -1.40e-4 |
| 100 | 3.00e-2 | 4 | 9.62e-3 | 1.49e-2 | 1.15e-2 | 1.02e-2 | 54 | -2.55e-2 | +7.21e-3 | -6.84e-3 | -2.17e-3 |
| 101 | 3.00e-2 | 6 | 1.00e-2 | 1.31e-2 | 1.13e-2 | 1.13e-2 | 50 | -6.16e-3 | +3.01e-3 | +1.68e-4 | -1.09e-3 |
| 102 | 3.00e-2 | 1 | 1.23e-2 | 1.23e-2 | 1.23e-2 | 1.23e-2 | 50 | +1.81e-3 | +1.81e-3 | +1.81e-3 | -8.00e-4 |
| 103 | 3.00e-2 | 1 | 1.22e-2 | 1.22e-2 | 1.22e-2 | 1.22e-2 | 330 | -4.23e-5 | -4.23e-5 | -4.23e-5 | -7.24e-4 |
| 104 | 3.00e-2 | 1 | 3.03e-2 | 3.03e-2 | 3.03e-2 | 3.03e-2 | 360 | +2.54e-3 | +2.54e-3 | +2.54e-3 | -3.98e-4 |
| 105 | 3.00e-2 | 1 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 313 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -3.34e-4 |
| 106 | 3.00e-2 | 1 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 327 | -1.64e-4 | -1.64e-4 | -1.64e-4 | -3.17e-4 |
| 107 | 3.00e-2 | 1 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 291 | +2.55e-4 | +2.55e-4 | +2.55e-4 | -2.60e-4 |
| 109 | 3.00e-2 | 1 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 347 | -6.17e-5 | -6.17e-5 | -6.17e-5 | -2.40e-4 |
| 110 | 3.00e-2 | 1 | 3.57e-2 | 3.57e-2 | 3.57e-2 | 3.57e-2 | 290 | +3.00e-4 | +3.00e-4 | +3.00e-4 | -1.86e-4 |
| 111 | 3.00e-2 | 1 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 278 | -3.80e-5 | -3.80e-5 | -3.80e-5 | -1.71e-4 |
| 112 | 3.00e-2 | 1 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 311 | -3.08e-5 | -3.08e-5 | -3.08e-5 | -1.57e-4 |
| 113 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 326 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -1.24e-4 |
| 114 | 3.00e-2 | 1 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 310 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -9.21e-5 |
| 115 | 3.00e-2 | 1 | 3.92e-2 | 3.92e-2 | 3.92e-2 | 3.92e-2 | 340 | -3.68e-6 | -3.68e-6 | -3.68e-6 | -8.33e-5 |
| 116 | 3.00e-2 | 1 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 337 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -5.56e-5 |
| 118 | 3.00e-2 | 1 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 328 | +3.02e-5 | +3.02e-5 | +3.02e-5 | -4.71e-5 |
| 119 | 3.00e-2 | 1 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 284 | +4.40e-5 | +4.40e-5 | +4.40e-5 | -3.80e-5 |
| 120 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 291 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -4.57e-5 |
| 121 | 3.00e-2 | 1 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 294 | +1.20e-4 | +1.20e-4 | +1.20e-4 | -2.92e-5 |
| 122 | 3.00e-2 | 1 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 290 | -1.35e-5 | -1.35e-5 | -1.35e-5 | -2.76e-5 |
| 123 | 3.00e-2 | 1 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 325 | +9.39e-5 | +9.39e-5 | +9.39e-5 | -1.55e-5 |
| 124 | 3.00e-2 | 1 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 330 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +1.29e-6 |
| 126 | 3.00e-2 | 1 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 357 | +1.82e-4 | +1.82e-4 | +1.82e-4 | +1.93e-5 |
| 127 | 3.00e-2 | 1 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 345 | +7.85e-5 | +7.85e-5 | +7.85e-5 | +2.53e-5 |
| 128 | 3.00e-2 | 1 | 5.19e-2 | 5.19e-2 | 5.19e-2 | 5.19e-2 | 324 | +6.74e-5 | +6.74e-5 | +6.74e-5 | +2.95e-5 |
| 129 | 3.00e-2 | 1 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 323 | -1.02e-4 | -1.02e-4 | -1.02e-4 | +1.63e-5 |
| 130 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 332 | +1.78e-4 | +1.78e-4 | +1.78e-4 | +3.25e-5 |
| 132 | 3.00e-2 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 410 | +1.97e-5 | +1.97e-5 | +1.97e-5 | +3.12e-5 |
| 133 | 3.00e-2 | 1 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 317 | +2.85e-4 | +2.85e-4 | +2.85e-4 | +5.66e-5 |
| 134 | 3.00e-2 | 1 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 307 | -1.72e-4 | -1.72e-4 | -1.72e-4 | +3.38e-5 |
| 135 | 3.00e-2 | 1 | 5.34e-2 | 5.34e-2 | 5.34e-2 | 5.34e-2 | 300 | -1.48e-4 | -1.48e-4 | -1.48e-4 | +1.56e-5 |
| 136 | 3.00e-2 | 1 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 303 | +7.96e-5 | +7.96e-5 | +7.96e-5 | +2.20e-5 |
| 137 | 3.00e-2 | 1 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 317 | +1.35e-4 | +1.35e-4 | +1.35e-4 | +3.33e-5 |
| 139 | 3.00e-2 | 2 | 5.84e-2 | 6.43e-2 | 6.14e-2 | 6.43e-2 | 277 | +6.14e-5 | +3.46e-4 | +2.04e-4 | +6.71e-5 |
| 141 | 3.00e-2 | 1 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 396 | -3.08e-4 | -3.08e-4 | -3.08e-4 | +2.96e-5 |
| 142 | 3.00e-2 | 1 | 6.50e-2 | 6.50e-2 | 6.50e-2 | 6.50e-2 | 312 | +4.25e-4 | +4.25e-4 | +4.25e-4 | +6.91e-5 |
| 143 | 3.00e-2 | 1 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 330 | -1.42e-4 | -1.42e-4 | -1.42e-4 | +4.80e-5 |
| 144 | 3.00e-2 | 1 | 6.51e-2 | 6.51e-2 | 6.51e-2 | 6.51e-2 | 278 | +1.71e-4 | +1.71e-4 | +1.71e-4 | +6.03e-5 |
| 145 | 3.00e-2 | 1 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 261 | -3.89e-4 | -3.89e-4 | -3.89e-4 | +1.54e-5 |
| 146 | 3.00e-2 | 1 | 5.98e-2 | 5.98e-2 | 5.98e-2 | 5.98e-2 | 283 | +6.28e-5 | +6.28e-5 | +6.28e-5 | +2.02e-5 |
| 148 | 3.00e-2 | 1 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 365 | +2.93e-5 | +2.93e-5 | +2.93e-5 | +2.11e-5 |
| 149 | 3.00e-2 | 1 | 6.89e-2 | 6.89e-2 | 6.89e-2 | 6.89e-2 | 287 | +4.54e-4 | +4.54e-4 | +4.54e-4 | +6.43e-5 |
| 150 | 3.00e-3 | 1 | 6.44e-2 | 6.44e-2 | 6.44e-2 | 6.44e-2 | 312 | -2.17e-4 | -2.17e-4 | -2.17e-4 | +3.62e-5 |
| 151 | 3.00e-3 | 1 | 6.76e-2 | 6.76e-2 | 6.76e-2 | 6.76e-2 | 330 | +1.48e-4 | +1.48e-4 | +1.48e-4 | +4.74e-5 |
| 152 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 322 | -7.33e-3 | -7.33e-3 | -7.33e-3 | -6.91e-4 |
| 153 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 293 | -1.98e-4 | -1.98e-4 | -1.98e-4 | -6.41e-4 |
| 154 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 287 | -1.18e-5 | -1.18e-5 | -1.18e-5 | -5.79e-4 |
| 155 | 3.00e-3 | 1 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 280 | -9.21e-5 | -9.21e-5 | -9.21e-5 | -5.30e-4 |
| 157 | 3.00e-3 | 1 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 354 | -6.08e-5 | -6.08e-5 | -6.08e-5 | -4.83e-4 |
| 158 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 334 | +2.57e-4 | +2.57e-4 | +2.57e-4 | -4.09e-4 |
| 159 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 322 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -3.82e-4 |
| 160 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 318 | +5.43e-5 | +5.43e-5 | +5.43e-5 | -3.38e-4 |
| 161 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 322 | +3.57e-5 | +3.57e-5 | +3.57e-5 | -3.01e-4 |
| 163 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 378 | -3.46e-5 | -3.46e-5 | -3.46e-5 | -2.74e-4 |
| 164 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 281 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -2.16e-4 |
| 165 | 3.00e-3 | 1 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 280 | -3.88e-4 | -3.88e-4 | -3.88e-4 | -2.33e-4 |
| 166 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 310 | -4.59e-5 | -4.59e-5 | -4.59e-5 | -2.15e-4 |
| 167 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 344 | +1.50e-4 | +1.50e-4 | +1.50e-4 | -1.78e-4 |
| 168 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 334 | +6.93e-5 | +6.93e-5 | +6.93e-5 | -1.53e-4 |
| 170 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 361 | -2.75e-5 | -2.75e-5 | -2.75e-5 | -1.41e-4 |
| 171 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 323 | +8.67e-5 | +8.67e-5 | +8.67e-5 | -1.18e-4 |
| 172 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 290 | -7.61e-5 | -7.61e-5 | -7.61e-5 | -1.14e-4 |
| 173 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 336 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -1.16e-4 |
| 174 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 313 | +1.55e-4 | +1.55e-4 | +1.55e-4 | -8.90e-5 |
| 175 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 276 | -4.56e-5 | -4.56e-5 | -4.56e-5 | -8.46e-5 |
| 176 | 3.00e-3 | 1 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 313 | -1.51e-4 | -1.51e-4 | -1.51e-4 | -9.12e-5 |
| 178 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 356 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -7.03e-5 |
| 179 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 317 | +1.32e-4 | +1.32e-4 | +1.32e-4 | -5.01e-5 |
| 180 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 329 | +1.85e-5 | +1.85e-5 | +1.85e-5 | -4.32e-5 |
| 181 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 313 | -1.87e-5 | -1.87e-5 | -1.87e-5 | -4.08e-5 |
| 182 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 308 | -3.76e-5 | -3.76e-5 | -3.76e-5 | -4.05e-5 |
| 183 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 309 | -3.24e-5 | -3.24e-5 | -3.24e-5 | -3.97e-5 |
| 184 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 295 | -9.59e-5 | -9.59e-5 | -9.59e-5 | -4.53e-5 |
| 185 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 299 | +4.05e-5 | +4.05e-5 | +4.05e-5 | -3.67e-5 |
| 186 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 271 | +6.70e-5 | +6.70e-5 | +6.70e-5 | -2.63e-5 |
| 188 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 349 | -4.16e-5 | -4.16e-5 | -4.16e-5 | -2.79e-5 |
| 189 | 3.00e-3 | 1 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 303 | +2.30e-4 | +2.30e-4 | +2.30e-4 | -2.09e-6 |
| 190 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 321 | -1.26e-4 | -1.26e-4 | -1.26e-4 | -1.45e-5 |
| 191 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 312 | -2.14e-5 | -2.14e-5 | -2.14e-5 | -1.52e-5 |
| 192 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 280 | -5.48e-5 | -5.48e-5 | -5.48e-5 | -1.91e-5 |
| 193 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 317 | -7.08e-5 | -7.08e-5 | -7.08e-5 | -2.43e-5 |
| 194 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 288 | +1.96e-4 | +1.96e-4 | +1.96e-4 | -2.27e-6 |
| 196 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 383 | -9.14e-5 | -9.14e-5 | -9.14e-5 | -1.12e-5 |
| 197 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 314 | +3.53e-4 | +3.53e-4 | +3.53e-4 | +2.52e-5 |
| 198 | 3.00e-3 | 1 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 338 | -8.40e-5 | -8.40e-5 | -8.40e-5 | +1.43e-5 |
| 199 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 322 | +3.73e-5 | +3.73e-5 | +3.73e-5 | +1.66e-5 |

