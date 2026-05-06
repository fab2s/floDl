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
| nccl-async | 0.053730 | 0.9168 | +0.0043 | 2131.8 | 515 | 44.4 | 100% | 100% | 8.3 |

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
| nccl-async | 1.9533 | 0.7561 | 0.5708 | 0.4902 | 0.4616 | 0.5024 | 0.4903 | 0.4656 | 0.4777 | 0.4585 | 0.2017 | 0.1636 | 0.1471 | 0.1342 | 0.1254 | 0.0710 | 0.0627 | 0.0574 | 0.0554 | 0.0537 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3920 | 2.3 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3063 | 3.1 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.3017 | 3.0 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 390 | 386 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 2130.3 | 1.5 | epoch-boundary(199) |
| nccl-async | gpu1 | 2130.4 | 1.4 | epoch-boundary(199) |
| nccl-async | gpu0 | 2130.3 | 0.8 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.8s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu1 | 1.4s | 0.0s | 0.0s | 0.0s | 2.8s |
| resnet-graph | nccl-async | gpu2 | 1.5s | 0.0s | 0.0s | 0.0s | 3.1s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 160 | 0 | 515 | 44.4 | 1197/12150 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 209.3 | 9.8% |

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
| resnet-graph | nccl-async | 184 | 515 | 0 | 6.44e-3 | +6.61e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 515 | 9.20e-2 | 5.25e-2 | 0.00e0 | 3.64e-1 | 34.0 | -9.24e-5 | 1.91e-3 |
| resnet-graph | nccl-async | 1 | 515 | 9.42e-2 | 5.62e-2 | 0.00e0 | 4.41e-1 | 41.4 | -1.09e-4 | 2.87e-3 |
| resnet-graph | nccl-async | 2 | 515 | 9.36e-2 | 5.65e-2 | 0.00e0 | 4.88e-1 | 24.7 | -1.11e-4 | 2.90e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9881 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9868 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9976 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 63 (0,1,2,3,4,5,6,8…146,147) | 0 (—) | — | 0,1,2,3,4,5,6,8…146,147 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 22 | 22 |
| resnet-graph | nccl-async | 0e0 | 5 | 5 | 5 |
| resnet-graph | nccl-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 3 | 6 | 6 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 413 | +0.043 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 47 | +0.220 |
| resnet-graph | nccl-async | 3.00e-3 | 150–198 | 50 | +0.023 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 512 | +0.008 | 184 | +0.157 | +0.334 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 513 | 3.37e1–8.13e1 | 6.63e1 | 2.53e-3 | 3.74e-3 | 4.36e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 415 | 68–77756 | +1.278e-5 | 0.412 | +1.309e-5 | 0.434 | 91 | +1.600e-5 | 0.730 | 32–1055 | +1.070e-3 | 0.667 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 402 | 887–77756 | +1.292e-5 | 0.454 | +1.318e-5 | 0.472 | 90 | +1.601e-5 | 0.725 | 57–1055 | +1.065e-3 | 0.743 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 48 | 78617–116637 | +1.199e-5 | 0.094 | +1.211e-5 | 0.096 | 47 | +1.179e-5 | 0.087 | 710–958 | -1.836e-4 | 0.001 |
| resnet-graph | nccl-async | 3.00e-3 | 150–198 | 51 | 117613–155594 | -9.009e-6 | 0.049 | -9.115e-6 | 0.050 | 46 | -1.171e-5 | 0.072 | 628–976 | +2.193e-3 | 0.087 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.070e-3 | r0: +1.062e-3, r1: +1.074e-3, r2: +1.078e-3 | r0: 0.741, r1: 0.630, r2: 0.626 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.065e-3 | r0: +1.058e-3, r1: +1.069e-3, r2: +1.072e-3 | r0: 0.817, r1: 0.700, r2: 0.701 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | -1.836e-4 | r0: -1.363e-4, r1: -1.742e-4, r2: -2.387e-4 | r0: 0.000, r1: 0.000, r2: 0.001 | 1.75× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–198 | +2.193e-3 | r0: +2.260e-3, r1: +2.170e-3, r2: +2.151e-3 | r0: 0.093, r1: 0.085, r2: 0.084 | 1.05× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇▇▇▇███████████▅▄▄▄▅▅▅▅▅▅▅▆▁▁▁▁▁▁▁▁▁▁▁` | `▁█▇▇▇▇▇██▇▇▇▇██████████▆▆▇▇████████▅▆▇▇███████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 14 | 0.00e0 | 4.88e-1 | 1.03e-1 | 6.78e-2 | 22 | -5.57e-2 | +5.88e-3 | -1.13e-2 | -6.28e-3 |
| 1 | 3.00e-1 | 14 | 6.41e-2 | 1.08e-1 | 7.40e-2 | 8.35e-2 | 30 | -2.14e-2 | +2.39e-2 | +8.34e-4 | -2.85e-4 |
| 2 | 3.00e-1 | 6 | 7.05e-2 | 1.28e-1 | 8.73e-2 | 7.93e-2 | 27 | -1.87e-2 | +1.77e-2 | -3.02e-4 | -4.04e-4 |
| 3 | 3.00e-1 | 11 | 7.45e-2 | 1.45e-1 | 8.82e-2 | 8.05e-2 | 20 | -2.05e-2 | +2.06e-2 | +8.66e-5 | -2.51e-4 |
| 4 | 3.00e-1 | 9 | 7.79e-2 | 1.30e-1 | 9.41e-2 | 9.22e-2 | 29 | -1.90e-2 | +2.34e-2 | +8.52e-4 | +1.97e-4 |
| 5 | 3.00e-1 | 12 | 8.25e-2 | 1.36e-1 | 9.51e-2 | 9.82e-2 | 29 | -1.28e-2 | +1.34e-2 | +1.97e-5 | +1.74e-4 |
| 6 | 3.00e-1 | 6 | 8.66e-2 | 1.43e-1 | 1.01e-1 | 9.77e-2 | 30 | -1.56e-2 | +1.30e-2 | -2.38e-4 | -3.36e-5 |
| 7 | 3.00e-1 | 8 | 8.83e-2 | 1.34e-1 | 9.97e-2 | 9.10e-2 | 29 | -1.44e-2 | +1.10e-2 | -2.93e-4 | -2.82e-4 |
| 8 | 3.00e-1 | 8 | 8.24e-2 | 1.38e-1 | 9.75e-2 | 9.76e-2 | 31 | -1.39e-2 | +1.42e-2 | +3.30e-4 | +5.40e-5 |
| 9 | 3.00e-1 | 8 | 9.23e-2 | 1.53e-1 | 1.03e-1 | 9.23e-2 | 31 | -1.49e-2 | +1.53e-2 | -1.54e-4 | -1.95e-4 |
| 10 | 3.00e-1 | 7 | 9.67e-2 | 1.39e-1 | 1.04e-1 | 9.93e-2 | 33 | -1.06e-2 | +1.08e-2 | +1.73e-4 | -8.88e-5 |
| 11 | 3.00e-1 | 8 | 9.28e-2 | 1.36e-1 | 1.01e-1 | 9.28e-2 | 29 | -1.10e-2 | +1.02e-2 | -2.24e-4 | -2.46e-4 |
| 12 | 3.00e-1 | 12 | 8.25e-2 | 1.36e-1 | 9.30e-2 | 8.53e-2 | 29 | -1.51e-2 | +1.39e-2 | -2.12e-4 | -3.28e-4 |
| 13 | 3.00e-1 | 6 | 9.17e-2 | 1.44e-1 | 1.04e-1 | 9.34e-2 | 27 | -9.58e-3 | +1.20e-2 | -9.00e-5 | -3.51e-4 |
| 14 | 3.00e-1 | 8 | 8.17e-2 | 1.37e-1 | 9.46e-2 | 9.19e-2 | 31 | -1.55e-2 | +1.58e-2 | +1.83e-4 | -9.31e-5 |
| 15 | 3.00e-1 | 8 | 8.12e-2 | 1.45e-1 | 1.05e-1 | 9.59e-2 | 38 | -1.98e-2 | +1.55e-2 | -2.92e-4 | -3.47e-4 |
| 16 | 3.00e-1 | 5 | 1.02e-1 | 1.41e-1 | 1.11e-1 | 1.04e-1 | 43 | -7.65e-3 | +8.09e-3 | +2.65e-4 | -1.79e-4 |
| 17 | 3.00e-1 | 7 | 1.08e-1 | 1.44e-1 | 1.16e-1 | 1.12e-1 | 37 | -6.27e-3 | +5.46e-3 | +1.10e-4 | -8.83e-5 |
| 18 | 3.00e-1 | 4 | 9.92e-2 | 1.40e-1 | 1.13e-1 | 1.07e-1 | 47 | -8.17e-3 | +7.61e-3 | +4.25e-5 | -7.98e-5 |
| 19 | 3.00e-1 | 6 | 9.67e-2 | 1.51e-1 | 1.11e-1 | 1.05e-1 | 41 | -8.88e-3 | +7.13e-3 | -4.52e-4 | -2.71e-4 |
| 20 | 3.00e-1 | 6 | 1.00e-1 | 1.44e-1 | 1.10e-1 | 1.01e-1 | 41 | -8.76e-3 | +8.89e-3 | -6.30e-5 | -2.65e-4 |
| 21 | 3.00e-1 | 10 | 9.20e-2 | 1.51e-1 | 1.08e-1 | 1.03e-1 | 30 | -1.00e-2 | +8.10e-3 | -3.48e-5 | -1.10e-4 |
| 22 | 3.00e-1 | 4 | 8.74e-2 | 1.41e-1 | 1.04e-1 | 9.50e-2 | 36 | -1.37e-2 | +1.27e-2 | +3.22e-5 | -1.31e-4 |
| 23 | 3.00e-1 | 7 | 9.71e-2 | 1.41e-1 | 1.06e-1 | 9.78e-2 | 34 | -9.89e-3 | +9.40e-3 | +2.38e-5 | -1.49e-4 |
| 24 | 3.00e-1 | 7 | 8.46e-2 | 1.44e-1 | 1.02e-1 | 9.79e-2 | 34 | -1.02e-2 | +1.06e-2 | -2.48e-4 | -2.12e-4 |
| 25 | 3.00e-1 | 7 | 8.67e-2 | 1.49e-1 | 1.01e-1 | 9.76e-2 | 35 | -1.75e-2 | +1.43e-2 | +3.79e-5 | -9.29e-5 |
| 26 | 3.00e-1 | 8 | 7.99e-2 | 1.45e-1 | 9.88e-2 | 1.01e-1 | 34 | -2.12e-2 | +1.61e-2 | +2.25e-4 | +1.04e-4 |
| 27 | 3.00e-1 | 11 | 8.88e-2 | 1.50e-1 | 9.99e-2 | 9.21e-2 | 28 | -1.38e-2 | +1.35e-2 | -1.44e-4 | -1.40e-4 |
| 28 | 3.00e-1 | 5 | 8.56e-2 | 1.44e-1 | 1.00e-1 | 9.68e-2 | 32 | -1.73e-2 | +1.59e-2 | +3.94e-4 | +2.78e-5 |
| 29 | 3.00e-1 | 7 | 8.93e-2 | 1.39e-1 | 1.02e-1 | 1.03e-1 | 39 | -1.36e-2 | +1.12e-2 | +2.46e-4 | +1.42e-4 |
| 30 | 3.00e-1 | 8 | 9.24e-2 | 1.42e-1 | 1.02e-1 | 9.24e-2 | 32 | -1.27e-2 | +8.85e-3 | -5.12e-4 | -2.74e-4 |
| 31 | 3.00e-1 | 12 | 8.20e-2 | 1.43e-1 | 9.49e-2 | 9.45e-2 | 26 | -1.58e-2 | +1.10e-2 | -2.45e-4 | -1.32e-4 |
| 32 | 3.00e-1 | 6 | 8.12e-2 | 1.48e-1 | 9.72e-2 | 9.33e-2 | 28 | -2.15e-2 | +1.93e-2 | -1.97e-4 | -1.96e-4 |
| 33 | 3.00e-1 | 8 | 8.17e-2 | 1.51e-1 | 9.64e-2 | 9.33e-2 | 29 | -1.77e-2 | +1.88e-2 | +3.57e-4 | +6.41e-5 |
| 34 | 3.00e-1 | 9 | 7.96e-2 | 1.43e-1 | 9.45e-2 | 9.01e-2 | 32 | -1.66e-2 | +1.60e-2 | +5.25e-6 | -1.50e-5 |
| 35 | 3.00e-1 | 9 | 8.04e-2 | 1.47e-1 | 9.36e-2 | 9.47e-2 | 25 | -2.32e-2 | +1.67e-2 | +5.61e-5 | +1.33e-4 |
| 36 | 3.00e-1 | 9 | 7.83e-2 | 1.49e-1 | 9.53e-2 | 8.47e-2 | 23 | -2.47e-2 | +2.07e-2 | -2.80e-4 | -3.11e-4 |
| 37 | 3.00e-1 | 10 | 7.47e-2 | 1.43e-1 | 8.98e-2 | 7.95e-2 | 24 | -2.27e-2 | +2.41e-2 | -2.17e-4 | -4.70e-4 |
| 38 | 3.00e-1 | 13 | 7.67e-2 | 1.39e-1 | 9.09e-2 | 9.71e-2 | 24 | -2.18e-2 | +2.08e-2 | +4.60e-4 | +2.84e-4 |
| 39 | 3.00e-1 | 6 | 7.42e-2 | 1.49e-1 | 9.39e-2 | 8.31e-2 | 23 | -3.02e-2 | +2.36e-2 | -5.10e-4 | -1.99e-4 |
| 40 | 3.00e-1 | 9 | 8.18e-2 | 1.49e-1 | 9.58e-2 | 9.03e-2 | 27 | -2.19e-2 | +2.30e-2 | +3.82e-4 | -2.23e-5 |
| 41 | 3.00e-1 | 9 | 8.08e-2 | 1.42e-1 | 9.64e-2 | 9.86e-2 | 33 | -1.56e-2 | +1.40e-2 | +1.14e-4 | +1.22e-4 |
| 42 | 3.00e-1 | 13 | 7.85e-2 | 1.53e-1 | 9.53e-2 | 8.91e-2 | 27 | -1.89e-2 | +1.14e-2 | -5.54e-4 | -2.44e-4 |
| 43 | 3.00e-1 | 5 | 8.87e-2 | 1.49e-1 | 1.04e-1 | 8.89e-2 | 28 | -1.36e-2 | +1.43e-2 | -8.02e-4 | -6.34e-4 |
| 44 | 3.00e-1 | 1 | 8.99e-2 | 8.99e-2 | 8.99e-2 | 8.99e-2 | 28 | +3.70e-4 | +3.70e-4 | +3.70e-4 | -5.34e-4 |
| 45 | 3.00e-1 | 1 | 9.13e-2 | 9.13e-2 | 9.13e-2 | 9.13e-2 | 310 | +5.22e-5 | +5.22e-5 | +5.22e-5 | -4.75e-4 |
| 46 | 3.00e-1 | 1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 319 | +3.09e-3 | +3.09e-3 | +3.09e-3 | -1.18e-4 |
| 47 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 331 | -2.05e-4 | -2.05e-4 | -2.05e-4 | -1.27e-4 |
| 48 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 330 | +9.39e-7 | +9.39e-7 | +9.39e-7 | -1.14e-4 |
| 49 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 328 | -6.95e-5 | -6.95e-5 | -6.95e-5 | -1.10e-4 |
| 51 | 3.00e-1 | 2 | 2.25e-1 | 2.37e-1 | 2.31e-1 | 2.37e-1 | 268 | +1.64e-5 | +1.88e-4 | +1.02e-4 | -6.87e-5 |
| 53 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 354 | -3.53e-4 | -3.53e-4 | -3.53e-4 | -9.71e-5 |
| 54 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 317 | +2.57e-4 | +2.57e-4 | +2.57e-4 | -6.18e-5 |
| 55 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 298 | -1.12e-4 | -1.12e-4 | -1.12e-4 | -6.68e-5 |
| 56 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 287 | -1.08e-5 | -1.08e-5 | -1.08e-5 | -6.12e-5 |
| 57 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 304 | -7.12e-5 | -7.12e-5 | -7.12e-5 | -6.22e-5 |
| 58 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 307 | +1.32e-4 | +1.32e-4 | +1.32e-4 | -4.28e-5 |
| 60 | 3.00e-1 | 2 | 2.19e-1 | 2.26e-1 | 2.23e-1 | 2.26e-1 | 299 | -5.04e-5 | +1.08e-4 | +2.88e-5 | -2.84e-5 |
| 62 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 353 | -1.07e-4 | -1.07e-4 | -1.07e-4 | -3.62e-5 |
| 63 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 280 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -1.61e-5 |
| 64 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 302 | -2.22e-4 | -2.22e-4 | -2.22e-4 | -3.66e-5 |
| 65 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 300 | +3.24e-5 | +3.24e-5 | +3.24e-5 | -2.97e-5 |
| 66 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 286 | +8.17e-5 | +8.17e-5 | +8.17e-5 | -1.86e-5 |
| 67 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 283 | -2.78e-5 | -2.78e-5 | -2.78e-5 | -1.95e-5 |
| 68 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 330 | -7.59e-5 | -7.59e-5 | -7.59e-5 | -2.51e-5 |
| 69 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 300 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -7.39e-6 |
| 70 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 259 | -1.00e-4 | -1.00e-4 | -1.00e-4 | -1.67e-5 |
| 72 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 341 | -9.97e-5 | -9.97e-5 | -9.97e-5 | -2.50e-5 |
| 73 | 3.00e-1 | 2 | 2.10e-1 | 2.26e-1 | 2.18e-1 | 2.10e-1 | 247 | -3.10e-4 | +2.75e-4 | -1.77e-5 | -2.65e-5 |
| 75 | 3.00e-1 | 2 | 2.07e-1 | 2.30e-1 | 2.18e-1 | 2.30e-1 | 266 | -3.97e-5 | +3.90e-4 | +1.75e-4 | +1.40e-5 |
| 77 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 369 | -2.72e-4 | -2.72e-4 | -2.72e-4 | -1.46e-5 |
| 78 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 293 | +4.02e-4 | +4.02e-4 | +4.02e-4 | +2.70e-5 |
| 79 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 311 | -2.51e-4 | -2.51e-4 | -2.51e-4 | -7.88e-7 |
| 80 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 326 | +1.24e-4 | +1.24e-4 | +1.24e-4 | +1.17e-5 |
| 81 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 307 | +3.46e-6 | +3.46e-6 | +3.46e-6 | +1.09e-5 |
| 82 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 292 | -9.98e-5 | -9.98e-5 | -9.98e-5 | -2.10e-7 |
| 83 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 290 | -5.06e-5 | -5.06e-5 | -5.06e-5 | -5.25e-6 |
| 85 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 341 | -9.83e-6 | -9.83e-6 | -9.83e-6 | -5.71e-6 |
| 86 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 298 | +2.45e-4 | +2.45e-4 | +2.45e-4 | +1.94e-5 |
| 87 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 299 | -2.09e-4 | -2.09e-4 | -2.09e-4 | -3.44e-6 |
| 88 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 330 | -4.12e-6 | -4.12e-6 | -4.12e-6 | -3.51e-6 |
| 89 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 305 | +1.67e-4 | +1.67e-4 | +1.67e-4 | +1.36e-5 |
| 90 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 324 | -8.09e-5 | -8.09e-5 | -8.09e-5 | +4.12e-6 |
| 91 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 312 | +4.90e-5 | +4.90e-5 | +4.90e-5 | +8.61e-6 |
| 92 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 266 | -6.03e-5 | -6.03e-5 | -6.03e-5 | +1.72e-6 |
| 93 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 290 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -1.35e-5 |
| 94 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 297 | +6.44e-5 | +6.44e-5 | +6.44e-5 | -5.68e-6 |
| 96 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 363 | +1.67e-5 | +1.67e-5 | +1.67e-5 | -3.45e-6 |
| 97 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 320 | +2.00e-4 | +2.00e-4 | +2.00e-4 | +1.69e-5 |
| 98 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 310 | -1.05e-4 | -1.05e-4 | -1.05e-4 | +4.77e-6 |
| 99 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 294 | -6.29e-5 | -6.29e-5 | -6.29e-5 | -2.00e-6 |
| 100 | 3.00e-2 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 311 | -2.29e-5 | -2.29e-5 | -2.29e-5 | -4.08e-6 |
| 101 | 3.00e-2 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 279 | +5.43e-5 | +5.43e-5 | +5.43e-5 | +1.75e-6 |
| 102 | 3.00e-2 | 1 | 2.13e-2 | 2.13e-2 | 2.13e-2 | 2.13e-2 | 286 | -8.20e-3 | -8.20e-3 | -8.20e-3 | -8.19e-4 |
| 103 | 3.00e-2 | 1 | 2.20e-2 | 2.20e-2 | 2.20e-2 | 2.20e-2 | 310 | +9.57e-5 | +9.57e-5 | +9.57e-5 | -7.27e-4 |
| 104 | 3.00e-2 | 1 | 2.53e-2 | 2.53e-2 | 2.53e-2 | 2.53e-2 | 268 | +5.22e-4 | +5.22e-4 | +5.22e-4 | -6.02e-4 |
| 105 | 3.00e-2 | 1 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 265 | -6.97e-5 | -6.97e-5 | -6.97e-5 | -5.49e-4 |
| 106 | 3.00e-2 | 1 | 2.51e-2 | 2.51e-2 | 2.51e-2 | 2.51e-2 | 285 | +4.91e-5 | +4.91e-5 | +4.91e-5 | -4.89e-4 |
| 107 | 3.00e-2 | 1 | 2.66e-2 | 2.66e-2 | 2.66e-2 | 2.66e-2 | 277 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -4.20e-4 |
| 108 | 3.00e-2 | 1 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 287 | +9.70e-5 | +9.70e-5 | +9.70e-5 | -3.68e-4 |
| 109 | 3.00e-2 | 1 | 2.85e-2 | 2.85e-2 | 2.85e-2 | 2.85e-2 | 300 | +1.36e-4 | +1.36e-4 | +1.36e-4 | -3.18e-4 |
| 110 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 316 | +1.28e-4 | +1.28e-4 | +1.28e-4 | -2.73e-4 |
| 112 | 3.00e-2 | 1 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 356 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -2.25e-4 |
| 113 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 297 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -1.83e-4 |
| 114 | 3.00e-2 | 1 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 300 | -1.54e-4 | -1.54e-4 | -1.54e-4 | -1.80e-4 |
| 115 | 3.00e-2 | 1 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 325 | +5.86e-5 | +5.86e-5 | +5.86e-5 | -1.56e-4 |
| 116 | 3.00e-2 | 1 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 283 | +2.43e-4 | +2.43e-4 | +2.43e-4 | -1.16e-4 |
| 117 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 293 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -1.18e-4 |
| 118 | 3.00e-2 | 1 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 263 | +1.85e-4 | +1.85e-4 | +1.85e-4 | -8.81e-5 |
| 119 | 3.00e-2 | 1 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 300 | -2.14e-5 | -2.14e-5 | -2.14e-5 | -8.15e-5 |
| 120 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 260 | +2.17e-4 | +2.17e-4 | +2.17e-4 | -5.16e-5 |
| 121 | 3.00e-2 | 1 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 317 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -5.78e-5 |
| 122 | 3.00e-2 | 1 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 276 | +3.17e-4 | +3.17e-4 | +3.17e-4 | -2.03e-5 |
| 123 | 3.00e-2 | 1 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 303 | -7.61e-5 | -7.61e-5 | -7.61e-5 | -2.59e-5 |
| 124 | 3.00e-2 | 1 | 4.15e-2 | 4.15e-2 | 4.15e-2 | 4.15e-2 | 309 | +2.40e-4 | +2.40e-4 | +2.40e-4 | +6.63e-7 |
| 125 | 3.00e-2 | 1 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 270 | +9.89e-5 | +9.89e-5 | +9.89e-5 | +1.05e-5 |
| 126 | 3.00e-2 | 1 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 283 | -8.24e-5 | -8.24e-5 | -8.24e-5 | +1.20e-6 |
| 127 | 3.00e-2 | 1 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 260 | +6.78e-5 | +6.78e-5 | +6.78e-5 | +7.86e-6 |
| 128 | 3.00e-2 | 1 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 278 | -1.01e-4 | -1.01e-4 | -1.01e-4 | -3.06e-6 |
| 129 | 3.00e-2 | 1 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 285 | +1.47e-4 | +1.47e-4 | +1.47e-4 | +1.19e-5 |
| 130 | 3.00e-2 | 1 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 275 | +1.15e-4 | +1.15e-4 | +1.15e-4 | +2.22e-5 |
| 131 | 3.00e-2 | 1 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 261 | +2.80e-5 | +2.80e-5 | +2.80e-5 | +2.28e-5 |
| 132 | 3.00e-2 | 1 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 298 | +5.36e-5 | +5.36e-5 | +5.36e-5 | +2.59e-5 |
| 133 | 3.00e-2 | 1 | 4.89e-2 | 4.89e-2 | 4.89e-2 | 4.89e-2 | 317 | +2.39e-4 | +2.39e-4 | +2.39e-4 | +4.71e-5 |
| 134 | 3.00e-2 | 1 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 290 | +1.01e-4 | +1.01e-4 | +1.01e-4 | +5.25e-5 |
| 135 | 3.00e-2 | 1 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 292 | -3.31e-5 | -3.31e-5 | -3.31e-5 | +4.39e-5 |
| 137 | 3.00e-2 | 1 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 350 | +2.42e-6 | +2.42e-6 | +2.42e-6 | +3.98e-5 |
| 138 | 3.00e-2 | 1 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 325 | +2.91e-4 | +2.91e-4 | +2.91e-4 | +6.49e-5 |
| 139 | 3.00e-2 | 1 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 278 | -1.38e-5 | -1.38e-5 | -1.38e-5 | +5.70e-5 |
| 140 | 3.00e-2 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 289 | -2.09e-4 | -2.09e-4 | -2.09e-4 | +3.04e-5 |
| 141 | 3.00e-2 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 299 | +1.40e-4 | +1.40e-4 | +1.40e-4 | +4.14e-5 |
| 142 | 3.00e-2 | 1 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 296 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +4.99e-5 |
| 143 | 3.00e-2 | 1 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 245 | -3.08e-5 | -3.08e-5 | -3.08e-5 | +4.18e-5 |
| 144 | 3.00e-2 | 1 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 249 | -1.88e-4 | -1.88e-4 | -1.88e-4 | +1.89e-5 |
| 145 | 3.00e-2 | 1 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 270 | +7.04e-6 | +7.04e-6 | +7.04e-6 | +1.77e-5 |
| 146 | 3.00e-2 | 1 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 273 | +1.46e-4 | +1.46e-4 | +1.46e-4 | +3.05e-5 |
| 147 | 3.00e-2 | 1 | 5.71e-2 | 5.71e-2 | 5.71e-2 | 5.71e-2 | 262 | +1.40e-4 | +1.40e-4 | +1.40e-4 | +4.14e-5 |
| 148 | 3.00e-2 | 2 | 5.63e-2 | 6.17e-2 | 5.90e-2 | 6.17e-2 | 244 | -4.41e-5 | +3.80e-4 | +1.68e-4 | +6.76e-5 |
| 150 | 3.00e-3 | 1 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 373 | -2.36e-4 | -2.36e-4 | -2.36e-4 | +3.72e-5 |
| 151 | 3.00e-3 | 1 | 6.63e-2 | 6.63e-2 | 6.63e-2 | 6.63e-2 | 263 | +6.06e-4 | +6.06e-4 | +6.06e-4 | +9.41e-5 |
| 152 | 3.00e-3 | 1 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 272 | -9.03e-3 | -9.03e-3 | -9.03e-3 | -8.18e-4 |
| 153 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 265 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -7.50e-4 |
| 154 | 3.00e-3 | 1 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 264 | -8.62e-5 | -8.62e-5 | -8.62e-5 | -6.84e-4 |
| 155 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 286 | +1.47e-4 | +1.47e-4 | +1.47e-4 | -6.01e-4 |
| 156 | 3.00e-3 | 2 | 5.52e-3 | 5.67e-3 | 5.60e-3 | 5.52e-3 | 222 | -1.23e-4 | +5.57e-5 | -3.38e-5 | -4.94e-4 |
| 158 | 3.00e-3 | 2 | 5.21e-3 | 5.71e-3 | 5.46e-3 | 5.71e-3 | 223 | -1.91e-4 | +4.12e-4 | +1.10e-4 | -3.76e-4 |
| 160 | 3.00e-3 | 2 | 5.43e-3 | 5.89e-3 | 5.66e-3 | 5.89e-3 | 222 | -1.65e-4 | +3.70e-4 | +1.03e-4 | -2.82e-4 |
| 161 | 3.00e-3 | 1 | 5.08e-3 | 5.08e-3 | 5.08e-3 | 5.08e-3 | 270 | -5.51e-4 | -5.51e-4 | -5.51e-4 | -3.09e-4 |
| 162 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 260 | +4.55e-4 | +4.55e-4 | +4.55e-4 | -2.33e-4 |
| 163 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 265 | -3.48e-5 | -3.48e-5 | -3.48e-5 | -2.13e-4 |
| 164 | 3.00e-3 | 1 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 305 | +5.95e-5 | +5.95e-5 | +5.95e-5 | -1.86e-4 |
| 165 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 287 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -1.50e-4 |
| 167 | 3.00e-3 | 2 | 5.85e-3 | 6.39e-3 | 6.12e-3 | 6.39e-3 | 218 | -9.19e-5 | +4.06e-4 | +1.57e-4 | -8.95e-5 |
| 168 | 3.00e-3 | 1 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 274 | -5.33e-4 | -5.33e-4 | -5.33e-4 | -1.34e-4 |
| 169 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 264 | +1.70e-4 | +1.70e-4 | +1.70e-4 | -1.03e-4 |
| 170 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 262 | -4.33e-5 | -4.33e-5 | -4.33e-5 | -9.74e-5 |
| 171 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 305 | -6.86e-5 | -6.86e-5 | -6.86e-5 | -9.46e-5 |
| 172 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 268 | +4.59e-4 | +4.59e-4 | +4.59e-4 | -3.92e-5 |
| 173 | 3.00e-3 | 1 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 295 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -5.30e-5 |
| 174 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 290 | +7.46e-5 | +7.46e-5 | +7.46e-5 | -4.03e-5 |
| 175 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 258 | +4.31e-5 | +4.31e-5 | +4.31e-5 | -3.19e-5 |
| 176 | 3.00e-3 | 1 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 250 | -1.85e-4 | -1.85e-4 | -1.85e-4 | -4.72e-5 |
| 177 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 260 | -8.46e-5 | -8.46e-5 | -8.46e-5 | -5.09e-5 |
| 178 | 3.00e-3 | 1 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 275 | +1.72e-4 | +1.72e-4 | +1.72e-4 | -2.87e-5 |
| 179 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 257 | +5.97e-5 | +5.97e-5 | +5.97e-5 | -1.98e-5 |
| 180 | 3.00e-3 | 1 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 255 | -1.95e-4 | -1.95e-4 | -1.95e-4 | -3.73e-5 |
| 181 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 260 | +4.77e-5 | +4.77e-5 | +4.77e-5 | -2.88e-5 |
| 182 | 3.00e-3 | 1 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 270 | +5.79e-5 | +5.79e-5 | +5.79e-5 | -2.02e-5 |
| 183 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 287 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -1.79e-6 |
| 184 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 274 | +1.30e-4 | +1.30e-4 | +1.30e-4 | +1.14e-5 |
| 185 | 3.00e-3 | 1 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 304 | -2.58e-5 | -2.58e-5 | -2.58e-5 | +7.69e-6 |
| 186 | 3.00e-3 | 1 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 270 | +6.92e-5 | +6.92e-5 | +6.92e-5 | +1.38e-5 |
| 187 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 278 | -2.86e-4 | -2.86e-4 | -2.86e-4 | -1.62e-5 |
| 188 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 296 | +5.37e-5 | +5.37e-5 | +5.37e-5 | -9.20e-6 |
| 189 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 296 | +9.67e-5 | +9.67e-5 | +9.67e-5 | +1.39e-6 |
| 190 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 264 | -3.18e-5 | -3.18e-5 | -3.18e-5 | -1.92e-6 |
| 191 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 270 | -5.73e-5 | -5.73e-5 | -5.73e-5 | -7.46e-6 |
| 192 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 282 | +1.56e-4 | +1.56e-4 | +1.56e-4 | +8.90e-6 |
| 193 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 263 | -5.22e-5 | -5.22e-5 | -5.22e-5 | +2.79e-6 |
| 194 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 306 | -5.86e-5 | -5.86e-5 | -5.86e-5 | -3.35e-6 |
| 195 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 257 | +3.08e-4 | +3.08e-4 | +3.08e-4 | +2.78e-5 |
| 196 | 3.00e-3 | 1 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 266 | -3.68e-4 | -3.68e-4 | -3.68e-4 | -1.18e-5 |
| 197 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 258 | +5.84e-5 | +5.84e-5 | +5.84e-5 | -4.76e-6 |
| 198 | 3.00e-3 | 2 | 6.18e-3 | 6.44e-3 | 6.31e-3 | 6.44e-3 | 238 | -7.39e-5 | +1.71e-4 | +4.86e-5 | +6.61e-6 |

