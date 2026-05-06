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
| nccl-async | 0.064528 | 0.9169 | +0.0044 | 2091.2 | 387 | 48.1 | 100% | 100% | 6.3 |

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
| nccl-async | 1.9498 | 0.7484 | 0.5621 | 0.5089 | 0.5210 | 0.5146 | 0.4925 | 0.4771 | 0.4723 | 0.4627 | 0.2125 | 0.1777 | 0.1618 | 0.1521 | 0.1462 | 0.0826 | 0.0725 | 0.0712 | 0.0648 | 0.0645 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3926 | 2.4 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3103 | 3.2 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2971 | 3.1 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 388 | 383 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 2089.7 | 1.5 | epoch-boundary(199) |
| nccl-async | gpu2 | 2089.7 | 1.5 | epoch-boundary(199) |
| nccl-async | gpu0 | 2089.7 | 0.7 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.7s | 0.0s | 0.0s | 0.0s | 2.0s |
| resnet-graph | nccl-async | gpu1 | 1.5s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.5s | 0.0s | 0.0s | 0.0s | 2.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 17 | 0 | 387 | 48.1 | 1691/12678 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 178.8 | 8.6% |

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
| resnet-graph | nccl-async | 181 | 387 | 0 | 7.71e-3 | +2.63e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 387 | 9.97e-2 | 6.01e-2 | 0.00e0 | 5.06e-1 | 28.4 | -1.13e-4 | 1.56e-3 |
| resnet-graph | nccl-async | 1 | 387 | 1.02e-1 | 6.09e-2 | 0.00e0 | 4.13e-1 | 48.6 | -1.03e-4 | 1.95e-3 |
| resnet-graph | nccl-async | 2 | 387 | 1.01e-1 | 6.05e-2 | 0.00e0 | 4.09e-1 | 23.0 | -1.04e-4 | 1.97e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9901 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9897 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9987 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 56 (0,1,2,3,4,6,8,9…149,151) | 0 (—) | — | 0,1,2,3,4,6,8,9…149,151 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 13 | 13 |
| resnet-graph | nccl-async | 0e0 | 5 | 6 | 6 |
| resnet-graph | nccl-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-4 | 5 | 0 | 0 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 294 | +0.091 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 45 | +0.180 |
| resnet-graph | nccl-async | 3.00e-3 | 151–199 | 43 | +0.281 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 384 | +0.014 | 180 | +0.190 | +0.389 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 385 | 3.39e1–8.00e1 | 6.55e1 | 3.13e-3 | 4.84e-3 | 6.47e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 296 | 73–77647 | +1.352e-5 | 0.565 | +1.382e-5 | 0.589 | 93 | +1.357e-5 | 0.753 | 39–1044 | +9.067e-4 | 0.671 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 284 | 920–77647 | +1.358e-5 | 0.660 | +1.384e-5 | 0.682 | 92 | +1.351e-5 | 0.746 | 62–1044 | +8.991e-4 | 0.790 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 46 | 78532–117148 | +1.067e-5 | 0.080 | +1.069e-5 | 0.080 | 45 | +1.051e-5 | 0.076 | 713–999 | +1.428e-3 | 0.033 |
| resnet-graph | nccl-async | 3.00e-3 | 151–199 | 44 | 118137–155840 | -2.775e-6 | 0.008 | -2.931e-6 | 0.009 | 43 | -2.921e-6 | 0.009 | 742–1027 | +1.643e-3 | 0.080 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +9.067e-4 | r0: +8.985e-4, r1: +9.097e-4, r2: +9.144e-4 | r0: 0.701, r1: 0.655, r2: 0.645 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +8.991e-4 | r0: +8.940e-4, r1: +9.017e-4, r2: +9.038e-4 | r0: 0.848, r1: 0.756, r2: 0.752 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.428e-3 | r0: +1.480e-3, r1: +1.388e-3, r2: +1.417e-3 | r0: 0.035, r1: 0.031, r2: 0.033 | 1.07× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 151–199 | +1.643e-3 | r0: +1.700e-3, r1: +1.647e-3, r2: +1.582e-3 | r0: 0.087, r1: 0.079, r2: 0.073 | 1.07× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇▇█████████████▇▄▄▄▅▅▅▅▅▅▅▄▁▁▁▁▁▁▁▁▁▁▂` | `▁▇▇▇█▇▇▇▇▇█████████████▇▆▇▇███████▇▆▇▇▇███████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 5.06e-1 | 1.13e-1 | 6.85e-2 | 24 | -4.77e-2 | +5.50e-3 | -9.71e-3 | -5.83e-3 |
| 1 | 3.00e-1 | 9 | 6.35e-2 | 1.16e-1 | 7.50e-2 | 7.78e-2 | 26 | -2.13e-2 | +1.52e-2 | -1.89e-4 | -1.67e-3 |
| 2 | 3.00e-1 | 7 | 7.08e-2 | 1.23e-1 | 9.43e-2 | 1.01e-1 | 42 | -8.87e-3 | +1.58e-2 | +1.26e-3 | -1.41e-4 |
| 3 | 3.00e-1 | 8 | 9.43e-2 | 1.25e-1 | 1.03e-1 | 1.02e-1 | 40 | -6.95e-3 | +6.28e-3 | +7.62e-5 | -2.36e-5 |
| 4 | 3.00e-1 | 5 | 9.67e-2 | 1.41e-1 | 1.09e-1 | 9.89e-2 | 33 | -7.91e-3 | +7.37e-3 | -4.68e-4 | -2.82e-4 |
| 5 | 3.00e-1 | 7 | 9.53e-2 | 1.33e-1 | 1.06e-1 | 1.08e-1 | 42 | -7.67e-3 | +8.94e-3 | +3.77e-4 | +4.04e-5 |
| 6 | 3.00e-1 | 5 | 1.02e-1 | 1.43e-1 | 1.14e-1 | 1.06e-1 | 37 | -7.30e-3 | +6.45e-3 | -2.82e-4 | -1.38e-4 |
| 7 | 3.00e-1 | 7 | 9.72e-2 | 1.48e-1 | 1.08e-1 | 9.95e-2 | 40 | -1.03e-2 | +8.92e-3 | -3.60e-4 | -3.05e-4 |
| 8 | 3.00e-1 | 6 | 1.00e-1 | 1.42e-1 | 1.09e-1 | 1.04e-1 | 39 | -9.39e-3 | +7.67e-3 | -7.13e-6 | -2.17e-4 |
| 9 | 3.00e-1 | 6 | 9.68e-2 | 1.44e-1 | 1.09e-1 | 1.05e-1 | 39 | -9.33e-3 | +7.72e-3 | -9.75e-5 | -1.80e-4 |
| 10 | 3.00e-1 | 7 | 1.02e-1 | 1.37e-1 | 1.08e-1 | 1.04e-1 | 41 | -6.77e-3 | +7.14e-3 | -6.39e-6 | -1.30e-4 |
| 11 | 3.00e-1 | 6 | 1.03e-1 | 1.41e-1 | 1.10e-1 | 1.04e-1 | 38 | -7.76e-3 | +7.38e-3 | -3.12e-6 | -1.24e-4 |
| 12 | 3.00e-1 | 6 | 9.71e-2 | 1.44e-1 | 1.07e-1 | 9.86e-2 | 36 | -1.06e-2 | +1.09e-2 | -5.25e-5 | -1.80e-4 |
| 13 | 3.00e-1 | 6 | 9.65e-2 | 1.40e-1 | 1.06e-1 | 1.05e-1 | 36 | -9.77e-3 | +9.78e-3 | +2.89e-4 | +1.44e-5 |
| 14 | 3.00e-1 | 7 | 9.19e-2 | 1.41e-1 | 1.06e-1 | 9.20e-2 | 29 | -7.63e-3 | +7.44e-3 | -8.06e-4 | -5.10e-4 |
| 15 | 3.00e-1 | 7 | 8.40e-2 | 1.37e-1 | 9.86e-2 | 9.95e-2 | 38 | -1.39e-2 | +1.50e-2 | +7.43e-4 | +8.06e-5 |
| 16 | 3.00e-1 | 6 | 9.81e-2 | 1.40e-1 | 1.09e-1 | 1.03e-1 | 37 | -7.45e-3 | +7.51e-3 | +7.65e-5 | +2.85e-5 |
| 17 | 3.00e-1 | 6 | 1.00e-1 | 1.35e-1 | 1.08e-1 | 1.03e-1 | 39 | -5.81e-3 | +6.70e-3 | +3.13e-5 | -1.99e-5 |
| 18 | 3.00e-1 | 6 | 9.60e-2 | 1.46e-1 | 1.09e-1 | 1.01e-1 | 40 | -1.08e-2 | +9.71e-3 | +1.07e-5 | -7.85e-5 |
| 19 | 3.00e-1 | 9 | 9.23e-2 | 1.46e-1 | 1.07e-1 | 1.04e-1 | 36 | -9.07e-3 | +8.19e-3 | +5.34e-5 | -1.82e-6 |
| 20 | 3.00e-1 | 4 | 9.45e-2 | 1.39e-1 | 1.08e-1 | 9.45e-2 | 40 | -9.28e-3 | +8.97e-3 | -6.82e-4 | -3.29e-4 |
| 21 | 3.00e-1 | 6 | 9.80e-2 | 1.39e-1 | 1.08e-1 | 1.03e-1 | 40 | -9.26e-3 | +8.76e-3 | +2.71e-4 | -1.24e-4 |
| 22 | 3.00e-1 | 6 | 1.01e-1 | 1.43e-1 | 1.11e-1 | 1.08e-1 | 43 | -8.68e-3 | +8.47e-3 | +2.06e-4 | -1.10e-5 |
| 23 | 3.00e-1 | 6 | 9.70e-2 | 1.44e-1 | 1.08e-1 | 1.05e-1 | 42 | -9.27e-3 | +8.80e-3 | -2.86e-6 | -2.09e-5 |
| 24 | 3.00e-1 | 6 | 1.01e-1 | 1.49e-1 | 1.11e-1 | 1.04e-1 | 37 | -8.73e-3 | +8.31e-3 | -1.12e-4 | -1.16e-4 |
| 25 | 3.00e-1 | 6 | 9.56e-2 | 1.43e-1 | 1.08e-1 | 1.06e-1 | 40 | -8.53e-3 | +9.75e-3 | +1.80e-4 | -9.23e-6 |
| 26 | 3.00e-1 | 5 | 9.80e-2 | 1.47e-1 | 1.10e-1 | 1.02e-1 | 40 | -1.04e-2 | +1.02e-2 | -6.40e-6 | -7.60e-5 |
| 27 | 3.00e-1 | 6 | 9.86e-2 | 1.49e-1 | 1.09e-1 | 1.00e-1 | 37 | -9.76e-3 | +7.45e-3 | -3.26e-4 | -2.40e-4 |
| 28 | 3.00e-1 | 8 | 9.92e-2 | 1.48e-1 | 1.08e-1 | 1.00e-1 | 40 | -9.96e-3 | +1.01e-2 | +7.44e-5 | -1.59e-4 |
| 29 | 3.00e-1 | 4 | 9.92e-2 | 1.42e-1 | 1.13e-1 | 9.92e-2 | 45 | -8.10e-3 | +7.41e-3 | -2.58e-4 | -2.84e-4 |
| 30 | 3.00e-1 | 8 | 1.01e-1 | 1.37e-1 | 1.11e-1 | 1.04e-1 | 44 | -6.30e-3 | +4.52e-3 | -9.55e-5 | -2.29e-4 |
| 31 | 3.00e-1 | 4 | 1.06e-1 | 1.45e-1 | 1.18e-1 | 1.11e-1 | 41 | -6.00e-3 | +7.01e-3 | +3.63e-4 | -8.17e-5 |
| 32 | 3.00e-1 | 8 | 9.43e-2 | 1.45e-1 | 1.10e-1 | 1.09e-1 | 40 | -1.07e-2 | +8.93e-3 | +1.65e-4 | +5.74e-5 |
| 33 | 3.00e-1 | 3 | 9.97e-2 | 1.47e-1 | 1.17e-1 | 9.97e-2 | 40 | -9.69e-3 | +8.82e-3 | -4.97e-4 | -1.84e-4 |
| 34 | 3.00e-1 | 6 | 9.88e-2 | 1.52e-1 | 1.11e-1 | 1.05e-1 | 46 | -9.13e-3 | +9.81e-3 | +1.23e-4 | -1.05e-4 |
| 35 | 3.00e-1 | 7 | 1.02e-1 | 1.46e-1 | 1.11e-1 | 1.02e-1 | 44 | -6.79e-3 | +5.71e-3 | -2.57e-4 | -2.33e-4 |
| 36 | 3.00e-1 | 4 | 1.08e-1 | 1.46e-1 | 1.18e-1 | 1.11e-1 | 50 | -6.40e-3 | +6.42e-3 | +3.06e-4 | -1.06e-4 |
| 37 | 3.00e-1 | 6 | 9.95e-2 | 1.45e-1 | 1.15e-1 | 9.95e-2 | 40 | -6.39e-3 | +5.18e-3 | -5.64e-4 | -4.14e-4 |
| 38 | 3.00e-1 | 1 | 1.02e-1 | 1.02e-1 | 1.02e-1 | 1.02e-1 | 50 | +4.89e-4 | +4.89e-4 | +4.89e-4 | -3.24e-4 |
| 39 | 3.00e-1 | 1 | 1.14e-1 | 1.14e-1 | 1.14e-1 | 1.14e-1 | 337 | +3.25e-4 | +3.25e-4 | +3.25e-4 | -2.59e-4 |
| 40 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 322 | +2.20e-3 | +2.20e-3 | +2.20e-3 | -1.35e-5 |
| 41 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 317 | -1.66e-4 | -1.66e-4 | -1.66e-4 | -2.87e-5 |
| 42 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 276 | -9.45e-5 | -9.45e-5 | -9.45e-5 | -3.53e-5 |
| 43 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 300 | -1.74e-4 | -1.74e-4 | -1.74e-4 | -4.91e-5 |
| 44 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 325 | +8.84e-5 | +8.84e-5 | +8.84e-5 | -3.54e-5 |
| 46 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 382 | +7.90e-5 | +7.90e-5 | +7.90e-5 | -2.39e-5 |
| 47 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 324 | +7.96e-5 | +7.96e-5 | +7.96e-5 | -1.36e-5 |
| 48 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 318 | -1.98e-4 | -1.98e-4 | -1.98e-4 | -3.20e-5 |
| 49 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 307 | +5.16e-5 | +5.16e-5 | +5.16e-5 | -2.37e-5 |
| 50 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 324 | -4.14e-5 | -4.14e-5 | -4.14e-5 | -2.54e-5 |
| 51 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 330 | +2.75e-5 | +2.75e-5 | +2.75e-5 | -2.01e-5 |
| 53 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 392 | +4.23e-5 | +4.23e-5 | +4.23e-5 | -1.39e-5 |
| 54 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 289 | +1.30e-4 | +1.30e-4 | +1.30e-4 | +5.35e-7 |
| 55 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 298 | -2.96e-4 | -2.96e-4 | -2.96e-4 | -2.91e-5 |
| 56 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 312 | +3.82e-5 | +3.82e-5 | +3.82e-5 | -2.23e-5 |
| 57 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 266 | +6.58e-5 | +6.58e-5 | +6.58e-5 | -1.35e-5 |
| 58 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 279 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -2.35e-5 |
| 59 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 306 | +5.48e-5 | +5.48e-5 | +5.48e-5 | -1.57e-5 |
| 60 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 326 | +6.21e-5 | +6.21e-5 | +6.21e-5 | -7.90e-6 |
| 61 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 320 | +5.79e-5 | +5.79e-5 | +5.79e-5 | -1.31e-6 |
| 62 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 313 | +1.43e-5 | +1.43e-5 | +1.43e-5 | +2.50e-7 |
| 64 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 390 | -4.89e-5 | -4.89e-5 | -4.89e-5 | -4.67e-6 |
| 65 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 300 | +2.08e-4 | +2.08e-4 | +2.08e-4 | +1.66e-5 |
| 66 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 273 | -3.00e-4 | -3.00e-4 | -3.00e-4 | -1.50e-5 |
| 67 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 292 | -6.36e-5 | -6.36e-5 | -6.36e-5 | -1.99e-5 |
| 68 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 297 | +9.87e-5 | +9.87e-5 | +9.87e-5 | -8.01e-6 |
| 69 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 307 | +5.95e-6 | +5.95e-6 | +5.95e-6 | -6.62e-6 |
| 70 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 345 | +1.02e-5 | +1.02e-5 | +1.02e-5 | -4.93e-6 |
| 71 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 327 | +9.99e-5 | +9.99e-5 | +9.99e-5 | +5.56e-6 |
| 73 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 366 | -6.01e-5 | -6.01e-5 | -6.01e-5 | -1.01e-6 |
| 74 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 313 | +1.22e-4 | +1.22e-4 | +1.22e-4 | +1.13e-5 |
| 75 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 286 | -1.95e-4 | -1.95e-4 | -1.95e-4 | -9.36e-6 |
| 76 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 315 | +2.61e-5 | +2.61e-5 | +2.61e-5 | -5.81e-6 |
| 77 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 293 | +2.94e-7 | +2.94e-7 | +2.94e-7 | -5.20e-6 |
| 78 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 295 | -7.13e-6 | -7.13e-6 | -7.13e-6 | -5.39e-6 |
| 79 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 295 | -3.49e-5 | -3.49e-5 | -3.49e-5 | -8.34e-6 |
| 80 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 327 | +3.58e-5 | +3.58e-5 | +3.58e-5 | -3.93e-6 |
| 82 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 370 | +6.01e-5 | +6.01e-5 | +6.01e-5 | +2.48e-6 |
| 83 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 293 | +8.23e-5 | +8.23e-5 | +8.23e-5 | +1.05e-5 |
| 84 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 304 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -1.09e-5 |
| 85 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 302 | +9.36e-5 | +9.36e-5 | +9.36e-5 | -4.49e-7 |
| 86 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 316 | -2.66e-5 | -2.66e-5 | -2.66e-5 | -3.07e-6 |
| 87 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 295 | +4.26e-5 | +4.26e-5 | +4.26e-5 | +1.50e-6 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 294 | -6.19e-5 | -6.19e-5 | -6.19e-5 | -4.83e-6 |
| 89 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 312 | -7.02e-5 | -7.02e-5 | -7.02e-5 | -1.14e-5 |
| 90 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 302 | +8.73e-5 | +8.73e-5 | +8.73e-5 | -1.50e-6 |
| 92 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 397 | -5.26e-6 | -5.26e-6 | -5.26e-6 | -1.88e-6 |
| 93 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 330 | +1.97e-4 | +1.97e-4 | +1.97e-4 | +1.80e-5 |
| 94 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 340 | -1.24e-4 | -1.24e-4 | -1.24e-4 | +3.77e-6 |
| 95 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 340 | +3.11e-5 | +3.11e-5 | +3.11e-5 | +6.50e-6 |
| 96 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 324 | -6.41e-7 | -6.41e-7 | -6.41e-7 | +5.79e-6 |
| 98 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 385 | -1.03e-5 | -1.03e-5 | -1.03e-5 | +4.18e-6 |
| 99 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 309 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +1.80e-5 |
| 100 | 3.00e-2 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 312 | -2.25e-4 | -2.25e-4 | -2.25e-4 | -6.27e-6 |
| 101 | 3.00e-2 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 333 | +1.50e-5 | +1.50e-5 | +1.50e-5 | -4.14e-6 |
| 102 | 3.00e-2 | 1 | 2.39e-2 | 2.39e-2 | 2.39e-2 | 2.39e-2 | 337 | -6.48e-3 | -6.48e-3 | -6.48e-3 | -6.51e-4 |
| 103 | 3.00e-2 | 1 | 2.46e-2 | 2.46e-2 | 2.46e-2 | 2.46e-2 | 307 | +9.61e-5 | +9.61e-5 | +9.61e-5 | -5.77e-4 |
| 104 | 3.00e-2 | 1 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 250 | +7.44e-5 | +7.44e-5 | +7.44e-5 | -5.11e-4 |
| 105 | 3.00e-2 | 1 | 2.41e-2 | 2.41e-2 | 2.41e-2 | 2.41e-2 | 299 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -4.74e-4 |
| 107 | 3.00e-2 | 1 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 352 | +2.65e-4 | +2.65e-4 | +2.65e-4 | -4.00e-4 |
| 108 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 293 | +4.00e-4 | +4.00e-4 | +4.00e-4 | -3.20e-4 |
| 109 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 291 | -1.52e-4 | -1.52e-4 | -1.52e-4 | -3.03e-4 |
| 110 | 3.00e-2 | 1 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 304 | +6.50e-5 | +6.50e-5 | +6.50e-5 | -2.66e-4 |
| 111 | 3.00e-2 | 1 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 319 | +1.49e-4 | +1.49e-4 | +1.49e-4 | -2.25e-4 |
| 112 | 3.00e-2 | 1 | 3.18e-2 | 3.18e-2 | 3.18e-2 | 3.18e-2 | 284 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -1.86e-4 |
| 113 | 3.00e-2 | 1 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 326 | -5.85e-5 | -5.85e-5 | -5.85e-5 | -1.73e-4 |
| 114 | 3.00e-2 | 1 | 3.40e-2 | 3.40e-2 | 3.40e-2 | 3.40e-2 | 314 | +2.66e-4 | +2.66e-4 | +2.66e-4 | -1.29e-4 |
| 115 | 3.00e-2 | 1 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 306 | -3.76e-5 | -3.76e-5 | -3.76e-5 | -1.20e-4 |
| 117 | 3.00e-2 | 1 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 370 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -9.73e-5 |
| 118 | 3.00e-2 | 1 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 327 | +2.51e-4 | +2.51e-4 | +2.51e-4 | -6.25e-5 |
| 119 | 3.00e-2 | 1 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 285 | -1.67e-5 | -1.67e-5 | -1.67e-5 | -5.79e-5 |
| 120 | 3.00e-2 | 1 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 284 | -1.47e-4 | -1.47e-4 | -1.47e-4 | -6.68e-5 |
| 121 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 318 | +6.71e-5 | +6.71e-5 | +6.71e-5 | -5.34e-5 |
| 122 | 3.00e-2 | 1 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 307 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -3.51e-5 |
| 123 | 3.00e-2 | 1 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 281 | +1.21e-4 | +1.21e-4 | +1.21e-4 | -1.95e-5 |
| 124 | 3.00e-2 | 1 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 298 | -5.72e-5 | -5.72e-5 | -5.72e-5 | -2.32e-5 |
| 125 | 3.00e-2 | 1 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 289 | +1.86e-4 | +1.86e-4 | +1.86e-4 | -2.35e-6 |
| 126 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 297 | -2.79e-5 | -2.79e-5 | -2.79e-5 | -4.90e-6 |
| 127 | 3.00e-2 | 1 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 290 | +1.17e-4 | +1.17e-4 | +1.17e-4 | +7.26e-6 |
| 128 | 3.00e-2 | 1 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 270 | +5.06e-6 | +5.06e-6 | +5.06e-6 | +7.04e-6 |
| 129 | 3.00e-2 | 1 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 289 | +3.53e-5 | +3.53e-5 | +3.53e-5 | +9.86e-6 |
| 131 | 3.00e-2 | 1 | 4.44e-2 | 4.44e-2 | 4.44e-2 | 4.44e-2 | 342 | +1.03e-4 | +1.03e-4 | +1.03e-4 | +1.92e-5 |
| 132 | 3.00e-2 | 1 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 315 | +2.53e-4 | +2.53e-4 | +2.53e-4 | +4.26e-5 |
| 133 | 3.00e-2 | 1 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 331 | -7.36e-5 | -7.36e-5 | -7.36e-5 | +3.09e-5 |
| 134 | 3.00e-2 | 1 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 360 | +1.59e-4 | +1.59e-4 | +1.59e-4 | +4.38e-5 |
| 135 | 3.00e-2 | 1 | 5.11e-2 | 5.11e-2 | 5.11e-2 | 5.11e-2 | 310 | +8.88e-5 | +8.88e-5 | +8.88e-5 | +4.83e-5 |
| 136 | 3.00e-2 | 1 | 4.86e-2 | 4.86e-2 | 4.86e-2 | 4.86e-2 | 292 | -1.69e-4 | -1.69e-4 | -1.69e-4 | +2.65e-5 |
| 137 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 287 | +8.37e-5 | +8.37e-5 | +8.37e-5 | +3.22e-5 |
| 138 | 3.00e-2 | 1 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 309 | +4.70e-5 | +4.70e-5 | +4.70e-5 | +3.37e-5 |
| 139 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 300 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +4.35e-5 |
| 141 | 3.00e-2 | 2 | 5.25e-2 | 5.46e-2 | 5.35e-2 | 5.46e-2 | 255 | -2.13e-6 | +1.50e-4 | +7.40e-5 | +5.01e-5 |
| 143 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 363 | -2.02e-4 | -2.02e-4 | -2.02e-4 | +2.49e-5 |
| 144 | 3.00e-2 | 1 | 5.81e-2 | 5.81e-2 | 5.81e-2 | 5.81e-2 | 280 | +4.88e-4 | +4.88e-4 | +4.88e-4 | +7.12e-5 |
| 145 | 3.00e-2 | 1 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 319 | -1.53e-4 | -1.53e-4 | -1.53e-4 | +4.87e-5 |
| 146 | 3.00e-2 | 1 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 300 | +5.32e-5 | +5.32e-5 | +5.32e-5 | +4.92e-5 |
| 147 | 3.00e-2 | 1 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 304 | -5.30e-6 | -5.30e-6 | -5.30e-6 | +4.37e-5 |
| 148 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 305 | +9.30e-5 | +9.30e-5 | +9.30e-5 | +4.87e-5 |
| 149 | 3.00e-2 | 1 | 5.91e-2 | 5.91e-2 | 5.91e-2 | 5.91e-2 | 316 | +7.40e-5 | +7.40e-5 | +7.40e-5 | +5.12e-5 |
| 151 | 3.00e-3 | 1 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 376 | +9.65e-5 | +9.65e-5 | +9.65e-5 | +5.57e-5 |
| 152 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 320 | -6.99e-3 | -6.99e-3 | -6.99e-3 | -6.49e-4 |
| 153 | 3.00e-3 | 1 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 305 | -3.97e-4 | -3.97e-4 | -3.97e-4 | -6.24e-4 |
| 154 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 310 | -5.12e-5 | -5.12e-5 | -5.12e-5 | -5.67e-4 |
| 155 | 3.00e-3 | 1 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 323 | +9.67e-5 | +9.67e-5 | +9.67e-5 | -5.00e-4 |
| 156 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 310 | +9.09e-5 | +9.09e-5 | +9.09e-5 | -4.41e-4 |
| 157 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 316 | -7.61e-5 | -7.61e-5 | -7.61e-5 | -4.05e-4 |
| 158 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 299 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -3.52e-4 |
| 160 | 3.00e-3 | 1 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 352 | -2.31e-5 | -2.31e-5 | -2.31e-5 | -3.19e-4 |
| 161 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 304 | +2.13e-4 | +2.13e-4 | +2.13e-4 | -2.66e-4 |
| 162 | 3.00e-3 | 1 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 307 | -2.35e-4 | -2.35e-4 | -2.35e-4 | -2.63e-4 |
| 163 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 329 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -2.24e-4 |
| 164 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 310 | +1.17e-5 | +1.17e-5 | +1.17e-5 | -2.01e-4 |
| 165 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 323 | -1.56e-5 | -1.56e-5 | -1.56e-5 | -1.82e-4 |
| 166 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 307 | +4.90e-5 | +4.90e-5 | +4.90e-5 | -1.59e-4 |
| 168 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 339 | -3.79e-5 | -3.79e-5 | -3.79e-5 | -1.47e-4 |
| 169 | 3.00e-3 | 1 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 331 | +1.91e-4 | +1.91e-4 | +1.91e-4 | -1.13e-4 |
| 170 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 306 | -6.33e-5 | -6.33e-5 | -6.33e-5 | -1.08e-4 |
| 171 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 324 | -2.48e-5 | -2.48e-5 | -2.48e-5 | -9.97e-5 |
| 172 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 329 | +5.51e-5 | +5.51e-5 | +5.51e-5 | -8.42e-5 |
| 173 | 3.00e-3 | 1 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 293 | +5.68e-5 | +5.68e-5 | +5.68e-5 | -7.01e-5 |
| 174 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 303 | -1.18e-4 | -1.18e-4 | -1.18e-4 | -7.49e-5 |
| 175 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 265 | +7.03e-5 | +7.03e-5 | +7.03e-5 | -6.04e-5 |
| 176 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 288 | -2.54e-4 | -2.54e-4 | -2.54e-4 | -7.97e-5 |
| 177 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 264 | +2.52e-4 | +2.52e-4 | +2.52e-4 | -4.66e-5 |
| 179 | 3.00e-3 | 2 | 6.31e-3 | 7.08e-3 | 6.69e-3 | 7.08e-3 | 267 | -1.26e-4 | +4.35e-4 | +1.55e-4 | -5.56e-6 |
| 181 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 380 | -2.76e-4 | -2.76e-4 | -2.76e-4 | -3.26e-5 |
| 182 | 3.00e-3 | 1 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 276 | +4.37e-4 | +4.37e-4 | +4.37e-4 | +1.44e-5 |
| 183 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 296 | -2.23e-4 | -2.23e-4 | -2.23e-4 | -9.30e-6 |
| 184 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 303 | -8.81e-5 | -8.81e-5 | -8.81e-5 | -1.72e-5 |
| 185 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 308 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -1.60e-6 |
| 186 | 3.00e-3 | 1 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 328 | +5.70e-5 | +5.70e-5 | +5.70e-5 | +4.26e-6 |
| 188 | 3.00e-3 | 1 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 367 | -3.69e-6 | -3.69e-6 | -3.69e-6 | +3.47e-6 |
| 189 | 3.00e-3 | 1 | 7.51e-3 | 7.51e-3 | 7.51e-3 | 7.51e-3 | 309 | +2.41e-4 | +2.41e-4 | +2.41e-4 | +2.73e-5 |
| 190 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 293 | -1.79e-4 | -1.79e-4 | -1.79e-4 | +6.68e-6 |
| 191 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 290 | -1.81e-4 | -1.81e-4 | -1.81e-4 | -1.21e-5 |
| 192 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 299 | +7.71e-5 | +7.71e-5 | +7.71e-5 | -3.14e-6 |
| 193 | 3.00e-3 | 1 | 6.87e-3 | 6.87e-3 | 6.87e-3 | 6.87e-3 | 324 | -2.00e-5 | -2.00e-5 | -2.00e-5 | -4.83e-6 |
| 194 | 3.00e-3 | 1 | 7.37e-3 | 7.37e-3 | 7.37e-3 | 7.37e-3 | 326 | +2.16e-4 | +2.16e-4 | +2.16e-4 | +1.72e-5 |
| 195 | 3.00e-3 | 1 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 290 | -1.05e-4 | -1.05e-4 | -1.05e-4 | +5.02e-6 |
| 196 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 302 | -1.80e-4 | -1.80e-4 | -1.80e-4 | -1.35e-5 |
| 198 | 3.00e-3 | 1 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 391 | +8.72e-5 | +8.72e-5 | +8.72e-5 | -3.41e-6 |
| 199 | 3.00e-3 | 1 | 7.71e-3 | 7.71e-3 | 7.71e-3 | 7.71e-3 | 327 | +2.94e-4 | +2.94e-4 | +2.94e-4 | +2.63e-5 |

