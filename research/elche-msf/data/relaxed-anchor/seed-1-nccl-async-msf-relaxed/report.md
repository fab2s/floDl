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
| nccl-async | 0.063781 | 0.9154 | +0.0029 | 2002.8 | 585 | 40.7 | 100% | 100% | 11.2 |

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
| nccl-async | 1.9844 | 0.7488 | 0.5720 | 0.5092 | 0.4897 | 0.4639 | 0.4601 | 0.4410 | 0.4727 | 0.4673 | 0.2130 | 0.1782 | 0.1619 | 0.1512 | 0.1444 | 0.0832 | 0.0750 | 0.0703 | 0.0688 | 0.0638 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4000 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3038 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2962 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 398 | 395 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 2001.6 | 1.1 | epoch-boundary(199) |
| nccl-async | gpu2 | 2001.7 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1602.7 | 0.8 | epoch-boundary(160) |
| nccl-async | gpu1 | 1602.7 | 0.7 | epoch-boundary(160) |
| nccl-async | gpu2 | 1602.7 | 0.7 | epoch-boundary(160) |
| nccl-async | gpu0 | 2001.7 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 1.4s | 0.0s | 0.0s | 0.0s | 3.3s |
| resnet-graph | nccl-async | gpu1 | 1.8s | 0.0s | 0.0s | 0.0s | 4.8s |
| resnet-graph | nccl-async | gpu2 | 1.7s | 0.0s | 0.0s | 0.0s | 3.1s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 25 | 0 | 585 | 40.7 | 1522/11607 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 178.4 | 8.9% |

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
| resnet-graph | nccl-async | 182 | 585 | 0 | 6.88e-3 | -8.89e-7 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 585 | 9.74e-2 | 4.05e-2 | 0.00e0 | 3.47e-1 | 42.9 | -7.09e-5 | 1.46e-3 |
| resnet-graph | nccl-async | 1 | 585 | 9.92e-2 | 4.35e-2 | 0.00e0 | 3.95e-1 | 38.5 | -7.72e-5 | 2.11e-3 |
| resnet-graph | nccl-async | 2 | 585 | 9.83e-2 | 4.37e-2 | 0.00e0 | 4.28e-1 | 18.6 | -7.98e-5 | 2.14e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9815 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9810 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9954 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 70 (0,1,3,4,5,6,7,8…137,150) | 0 (—) | — | 0,1,3,4,5,6,7,8…137,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 26 | 26 |
| resnet-graph | nccl-async | 0e0 | 5 | 11 | 11 |
| resnet-graph | nccl-async | 0e0 | 10 | 3 | 3 |
| resnet-graph | nccl-async | 1e-4 | 3 | 5 | 5 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 493 | +0.061 |
| resnet-graph | nccl-async | 3.00e-2 | 99–148 | 44 | +0.255 |
| resnet-graph | nccl-async | 3.00e-3 | 150–198 | 43 | +0.061 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 582 | +0.011 | 182 | +0.219 | +0.361 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 583 | 3.33e1–8.07e1 | 6.83e1 | 2.34e-3 | 3.47e-3 | 3.75e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 495 | 75–77101 | +6.181e-6 | 0.268 | +6.274e-6 | 0.290 | 97 | +1.023e-5 | 0.636 | 36–1049 | +8.721e-4 | 0.371 |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | 482 | 920–77101 | +5.917e-6 | 0.307 | +5.979e-6 | 0.330 | 96 | +1.019e-5 | 0.627 | 71–1049 | +8.519e-4 | 0.467 |
| resnet-graph | nccl-async | 3.00e-2 | 99–148 | 45 | 78009–116275 | +5.078e-6 | 0.016 | +5.026e-6 | 0.016 | 43 | +4.654e-6 | 0.013 | 740–1022 | +1.810e-3 | 0.078 |
| resnet-graph | nccl-async | 3.00e-3 | 150–198 | 44 | 117255–155396 | -1.060e-5 | 0.065 | -1.065e-5 | 0.065 | 42 | -3.820e-6 | 0.015 | 793–1063 | +5.850e-4 | 0.005 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | +8.721e-4 | r0: +8.800e-4, r1: +8.701e-4, r2: +8.689e-4 | r0: 0.457, r1: 0.331, r2: 0.326 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | +8.519e-4 | r0: +8.585e-4, r1: +8.491e-4, r2: +8.507e-4 | r0: 0.601, r1: 0.401, r2: 0.406 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 99–148 | +1.810e-3 | r0: +1.847e-3, r1: +1.826e-3, r2: +1.760e-3 | r0: 0.080, r1: 0.079, r2: 0.074 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–198 | +5.850e-4 | r0: +5.931e-4, r1: +6.156e-4, r2: +5.479e-4 | r0: 0.006, r1: 0.006, r2: 0.005 | 1.12× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█████▄▄▄▄▅▅▅▅▅▅▂▁▁▁▁▁▁▁▁▁▁` | `▁█▇▆▇▇▇▇▇▇▇▆▆▆▇▇▇▇▇▇█▇▇▇▇▅▆▇▇▇▇▇▇▇▇▆▆▆▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 14 | 0.00e0 | 4.28e-1 | 1.01e-1 | 6.80e-2 | 27 | -5.60e-2 | +1.84e-2 | -9.99e-3 | -5.31e-3 |
| 1 | 3.00e-1 | 12 | 7.05e-2 | 1.18e-1 | 8.12e-2 | 8.91e-2 | 26 | -1.82e-2 | +1.62e-2 | +6.12e-4 | -5.99e-4 |
| 2 | 3.00e-1 | 5 | 8.14e-2 | 1.19e-1 | 9.23e-2 | 8.66e-2 | 28 | -1.05e-2 | +1.27e-2 | +1.92e-5 | -4.23e-4 |
| 3 | 3.00e-1 | 11 | 8.36e-2 | 1.25e-1 | 9.38e-2 | 9.33e-2 | 30 | -1.39e-2 | +1.26e-2 | +1.65e-4 | -7.50e-5 |
| 4 | 3.00e-1 | 5 | 8.79e-2 | 1.41e-1 | 1.04e-1 | 9.75e-2 | 31 | -1.31e-2 | +1.31e-2 | +7.06e-4 | +1.55e-4 |
| 5 | 3.00e-1 | 9 | 9.34e-2 | 1.32e-1 | 1.02e-1 | 1.01e-1 | 33 | -1.11e-2 | +1.03e-2 | +1.16e-4 | +8.98e-5 |
| 6 | 3.00e-1 | 6 | 9.81e-2 | 1.42e-1 | 1.09e-1 | 1.11e-1 | 36 | -1.09e-2 | +1.08e-2 | +4.74e-4 | +2.45e-4 |
| 7 | 3.00e-1 | 8 | 8.93e-2 | 1.37e-1 | 1.03e-1 | 1.10e-1 | 37 | -1.43e-2 | +9.76e-3 | +2.34e-5 | +2.27e-4 |
| 8 | 3.00e-1 | 7 | 9.32e-2 | 1.49e-1 | 1.06e-1 | 1.03e-1 | 35 | -1.37e-2 | +1.22e-2 | -5.38e-6 | +8.01e-5 |
| 9 | 3.00e-1 | 7 | 9.06e-2 | 1.54e-1 | 1.04e-1 | 1.03e-1 | 41 | -1.59e-2 | +1.23e-2 | -3.36e-4 | -1.12e-4 |
| 10 | 3.00e-1 | 7 | 9.57e-2 | 1.42e-1 | 1.06e-1 | 1.01e-1 | 31 | -1.16e-2 | +7.82e-3 | -1.95e-4 | -1.68e-4 |
| 11 | 3.00e-1 | 9 | 9.05e-2 | 1.40e-1 | 1.03e-1 | 1.02e-1 | 33 | -1.16e-2 | +1.33e-2 | +3.16e-4 | +8.09e-5 |
| 12 | 3.00e-1 | 5 | 8.98e-2 | 1.44e-1 | 1.06e-1 | 9.20e-2 | 39 | -7.74e-3 | +9.17e-3 | -1.03e-3 | -4.78e-4 |
| 13 | 3.00e-1 | 7 | 9.00e-2 | 1.40e-1 | 1.01e-1 | 9.49e-2 | 35 | -1.12e-2 | +8.57e-3 | -2.48e-4 | -3.99e-4 |
| 14 | 3.00e-1 | 6 | 9.27e-2 | 1.46e-1 | 1.09e-1 | 9.92e-2 | 37 | -8.29e-3 | +1.10e-2 | +1.01e-4 | -2.82e-4 |
| 15 | 3.00e-1 | 9 | 9.73e-2 | 1.40e-1 | 1.06e-1 | 1.01e-1 | 40 | -9.35e-3 | +8.09e-3 | +2.99e-5 | -1.54e-4 |
| 16 | 3.00e-1 | 4 | 9.72e-2 | 1.36e-1 | 1.12e-1 | 1.03e-1 | 34 | -8.58e-3 | +5.69e-3 | -3.16e-5 | -1.66e-4 |
| 17 | 3.00e-1 | 7 | 8.95e-2 | 1.39e-1 | 1.05e-1 | 1.03e-1 | 45 | -1.18e-2 | +1.08e-2 | +2.52e-4 | +1.02e-5 |
| 18 | 3.00e-1 | 6 | 9.73e-2 | 1.43e-1 | 1.11e-1 | 9.73e-2 | 37 | -7.59e-3 | +5.64e-3 | -5.95e-4 | -3.55e-4 |
| 19 | 3.00e-1 | 7 | 9.77e-2 | 1.41e-1 | 1.08e-1 | 1.05e-1 | 41 | -8.82e-3 | +9.39e-3 | +2.76e-4 | -8.44e-5 |
| 20 | 3.00e-1 | 8 | 9.79e-2 | 1.45e-1 | 1.11e-1 | 1.02e-1 | 38 | -9.67e-3 | +8.45e-3 | -1.02e-4 | -1.85e-4 |
| 21 | 3.00e-1 | 4 | 9.94e-2 | 1.39e-1 | 1.11e-1 | 9.94e-2 | 41 | -8.73e-3 | +7.49e-3 | -2.65e-4 | -2.92e-4 |
| 22 | 3.00e-1 | 6 | 1.04e-1 | 1.44e-1 | 1.14e-1 | 1.11e-1 | 34 | -7.07e-3 | +6.71e-3 | +4.09e-4 | -1.75e-5 |
| 23 | 3.00e-1 | 6 | 8.91e-2 | 1.44e-1 | 1.07e-1 | 1.05e-1 | 41 | -1.06e-2 | +1.34e-2 | +3.97e-4 | +1.28e-4 |
| 24 | 3.00e-1 | 10 | 9.09e-2 | 1.46e-1 | 1.05e-1 | 9.41e-2 | 33 | -1.01e-2 | +7.77e-3 | -3.90e-4 | -2.66e-4 |
| 25 | 3.00e-1 | 4 | 9.62e-2 | 1.45e-1 | 1.12e-1 | 1.05e-1 | 36 | -8.86e-3 | +1.02e-2 | +6.63e-4 | -2.28e-5 |
| 26 | 3.00e-1 | 6 | 9.92e-2 | 1.46e-1 | 1.11e-1 | 1.08e-1 | 44 | -8.02e-3 | +9.60e-3 | +2.57e-4 | +6.00e-5 |
| 27 | 3.00e-1 | 6 | 1.04e-1 | 1.44e-1 | 1.13e-1 | 1.04e-1 | 42 | -7.01e-3 | +5.06e-3 | -4.16e-4 | -2.07e-4 |
| 28 | 3.00e-1 | 6 | 9.84e-2 | 1.37e-1 | 1.08e-1 | 1.00e-1 | 36 | -8.74e-3 | +6.59e-3 | -2.18e-4 | -2.62e-4 |
| 29 | 3.00e-1 | 7 | 9.34e-2 | 1.40e-1 | 1.05e-1 | 1.06e-1 | 38 | -1.19e-2 | +9.41e-3 | +1.59e-4 | -2.76e-5 |
| 30 | 3.00e-1 | 7 | 9.27e-2 | 1.44e-1 | 1.08e-1 | 9.27e-2 | 34 | -8.30e-3 | +8.94e-3 | -4.38e-4 | -3.81e-4 |
| 31 | 3.00e-1 | 6 | 9.27e-2 | 1.38e-1 | 1.07e-1 | 1.03e-1 | 40 | -1.02e-2 | +9.59e-3 | +5.02e-4 | -4.98e-5 |
| 32 | 3.00e-1 | 8 | 9.36e-2 | 1.45e-1 | 1.06e-1 | 1.15e-1 | 40 | -1.14e-2 | +9.02e-3 | +2.95e-4 | +2.17e-4 |
| 33 | 3.00e-1 | 5 | 9.54e-2 | 1.44e-1 | 1.11e-1 | 1.06e-1 | 41 | -1.02e-2 | +7.63e-3 | -1.56e-4 | +4.27e-5 |
| 34 | 3.00e-1 | 6 | 1.00e-1 | 1.47e-1 | 1.14e-1 | 1.08e-1 | 44 | -6.39e-3 | +7.47e-3 | -2.76e-5 | -3.84e-5 |
| 35 | 3.00e-1 | 8 | 1.06e-1 | 1.48e-1 | 1.14e-1 | 1.08e-1 | 41 | -6.76e-3 | +6.24e-3 | -5.28e-5 | -8.17e-5 |
| 36 | 3.00e-1 | 4 | 9.69e-2 | 1.55e-1 | 1.14e-1 | 9.98e-2 | 37 | -1.27e-2 | +1.03e-2 | -4.59e-4 | -2.99e-4 |
| 37 | 3.00e-1 | 7 | 9.37e-2 | 1.37e-1 | 1.06e-1 | 9.63e-2 | 36 | -7.36e-3 | +1.03e-2 | +4.97e-5 | -2.46e-4 |
| 38 | 3.00e-1 | 6 | 9.41e-2 | 1.43e-1 | 1.10e-1 | 1.03e-1 | 42 | -8.10e-3 | +1.07e-2 | +4.41e-4 | -3.74e-5 |
| 39 | 3.00e-1 | 7 | 1.06e-1 | 1.48e-1 | 1.17e-1 | 1.13e-1 | 49 | -6.76e-3 | +7.35e-3 | +2.37e-4 | +3.78e-5 |
| 40 | 3.00e-1 | 3 | 1.08e-1 | 1.55e-1 | 1.27e-1 | 1.08e-1 | 47 | -7.69e-3 | +5.54e-3 | -5.72e-4 | -2.08e-4 |
| 41 | 3.00e-1 | 6 | 1.05e-1 | 1.45e-1 | 1.15e-1 | 1.10e-1 | 46 | -6.97e-3 | +5.56e-3 | -1.17e-5 | -1.55e-4 |
| 42 | 3.00e-1 | 6 | 1.07e-1 | 1.47e-1 | 1.15e-1 | 1.08e-1 | 42 | -7.19e-3 | +5.49e-3 | -2.16e-4 | -2.20e-4 |
| 43 | 3.00e-1 | 6 | 1.03e-1 | 1.42e-1 | 1.12e-1 | 1.07e-1 | 48 | -7.27e-3 | +6.61e-3 | -9.69e-5 | -2.05e-4 |
| 44 | 3.00e-1 | 7 | 1.10e-1 | 1.53e-1 | 1.21e-1 | 1.19e-1 | 42 | -6.21e-3 | +6.74e-3 | +2.91e-4 | -1.05e-5 |
| 45 | 3.00e-1 | 5 | 9.22e-2 | 1.58e-1 | 1.10e-1 | 1.00e-1 | 36 | -1.53e-2 | +1.05e-2 | -9.26e-4 | -4.12e-4 |
| 46 | 3.00e-1 | 5 | 9.56e-2 | 1.45e-1 | 1.11e-1 | 9.56e-2 | 36 | -5.96e-3 | +7.68e-3 | -9.45e-4 | -7.46e-4 |
| 47 | 3.00e-1 | 7 | 9.71e-2 | 1.42e-1 | 1.10e-1 | 1.07e-1 | 46 | -8.15e-3 | +9.54e-3 | +3.89e-4 | -2.33e-4 |
| 48 | 3.00e-1 | 5 | 9.21e-2 | 1.44e-1 | 1.10e-1 | 1.03e-1 | 43 | -1.04e-2 | +7.09e-3 | -4.40e-4 | -3.49e-4 |
| 49 | 3.00e-1 | 6 | 9.63e-2 | 1.44e-1 | 1.13e-1 | 1.13e-1 | 48 | -9.75e-3 | +7.04e-3 | +2.15e-4 | -1.07e-4 |
| 50 | 3.00e-1 | 7 | 1.03e-1 | 1.57e-1 | 1.16e-1 | 1.08e-1 | 50 | -7.27e-3 | +6.27e-3 | -2.87e-4 | -2.35e-4 |
| 51 | 3.00e-1 | 4 | 1.09e-1 | 1.53e-1 | 1.23e-1 | 1.09e-1 | 53 | -7.40e-3 | +5.27e-3 | -2.73e-4 | -3.21e-4 |
| 52 | 3.00e-1 | 5 | 9.75e-2 | 1.55e-1 | 1.19e-1 | 1.06e-1 | 36 | -8.47e-3 | +4.66e-3 | -5.68e-4 | -4.57e-4 |
| 53 | 3.00e-1 | 8 | 9.77e-2 | 1.46e-1 | 1.08e-1 | 1.01e-1 | 41 | -7.46e-3 | +9.17e-3 | -1.08e-4 | -3.22e-4 |
| 54 | 3.00e-1 | 4 | 1.05e-1 | 1.47e-1 | 1.19e-1 | 1.05e-1 | 41 | -8.12e-3 | +5.83e-3 | -2.73e-4 | -3.91e-4 |
| 55 | 3.00e-1 | 7 | 1.04e-1 | 1.44e-1 | 1.13e-1 | 1.09e-1 | 43 | -7.43e-3 | +7.51e-3 | +1.72e-4 | -1.49e-4 |
| 56 | 3.00e-1 | 4 | 1.05e-1 | 1.47e-1 | 1.20e-1 | 1.14e-1 | 47 | -5.76e-3 | +7.23e-3 | +3.38e-4 | -3.06e-5 |
| 57 | 3.00e-1 | 6 | 1.02e-1 | 1.56e-1 | 1.18e-1 | 1.14e-1 | 49 | -1.05e-2 | +7.17e-3 | -9.12e-5 | -7.08e-5 |
| 58 | 3.00e-1 | 6 | 1.05e-1 | 1.52e-1 | 1.16e-1 | 1.07e-1 | 36 | -7.91e-3 | +6.95e-3 | -5.76e-5 | -1.08e-4 |
| 59 | 3.00e-1 | 6 | 9.92e-2 | 1.49e-1 | 1.10e-1 | 1.05e-1 | 39 | -1.01e-2 | +1.08e-2 | +2.34e-4 | -4.52e-6 |
| 60 | 3.00e-1 | 6 | 1.03e-1 | 1.42e-1 | 1.17e-1 | 1.19e-1 | 52 | -5.02e-3 | +6.83e-3 | +5.24e-4 | +2.10e-4 |
| 61 | 3.00e-1 | 4 | 1.18e-1 | 1.54e-1 | 1.30e-1 | 1.18e-1 | 47 | -3.66e-3 | +4.34e-3 | -3.03e-4 | -2.72e-5 |
| 62 | 3.00e-1 | 5 | 1.10e-1 | 1.50e-1 | 1.19e-1 | 1.11e-1 | 48 | -6.58e-3 | +5.82e-3 | -2.50e-4 | -1.60e-4 |
| 63 | 3.00e-1 | 5 | 1.09e-1 | 1.49e-1 | 1.21e-1 | 1.15e-1 | 51 | -6.11e-3 | +5.58e-3 | +1.67e-4 | -7.36e-5 |
| 64 | 3.00e-1 | 5 | 1.13e-1 | 1.53e-1 | 1.24e-1 | 1.17e-1 | 48 | -5.98e-3 | +4.55e-3 | -3.17e-5 | -9.79e-5 |
| 65 | 3.00e-1 | 5 | 1.07e-1 | 1.53e-1 | 1.21e-1 | 1.17e-1 | 41 | -6.56e-3 | +5.61e-3 | +2.67e-7 | -6.96e-5 |
| 66 | 3.00e-1 | 6 | 1.06e-1 | 1.49e-1 | 1.15e-1 | 1.12e-1 | 44 | -7.16e-3 | +8.38e-3 | +2.38e-4 | +4.36e-5 |
| 67 | 3.00e-1 | 7 | 1.05e-1 | 1.54e-1 | 1.17e-1 | 1.05e-1 | 41 | -6.45e-3 | +6.98e-3 | -3.01e-4 | -2.18e-4 |
| 68 | 3.00e-1 | 4 | 1.04e-1 | 1.44e-1 | 1.16e-1 | 1.13e-1 | 47 | -6.96e-3 | +7.32e-3 | +5.46e-4 | +8.75e-6 |
| 69 | 3.00e-1 | 7 | 1.09e-1 | 1.50e-1 | 1.19e-1 | 1.10e-1 | 42 | -4.69e-3 | +6.35e-3 | -7.33e-5 | -1.14e-4 |
| 70 | 3.00e-1 | 4 | 1.08e-1 | 1.44e-1 | 1.17e-1 | 1.08e-1 | 42 | -6.62e-3 | +5.92e-3 | -2.04e-4 | -2.01e-4 |
| 71 | 3.00e-1 | 5 | 1.06e-1 | 1.53e-1 | 1.19e-1 | 1.09e-1 | 43 | -5.90e-3 | +6.98e-3 | -2.11e-4 | -2.71e-4 |
| 72 | 3.00e-1 | 5 | 1.08e-1 | 1.49e-1 | 1.19e-1 | 1.17e-1 | 52 | -6.74e-3 | +6.97e-3 | +3.83e-4 | -4.05e-5 |
| 73 | 3.00e-1 | 5 | 1.04e-1 | 1.47e-1 | 1.18e-1 | 1.06e-1 | 47 | -7.03e-3 | +3.89e-3 | -5.83e-4 | -3.02e-4 |
| 74 | 3.00e-1 | 5 | 1.12e-1 | 1.53e-1 | 1.23e-1 | 1.16e-1 | 50 | -5.63e-3 | +6.30e-3 | +2.84e-4 | -1.24e-4 |
| 75 | 3.00e-1 | 5 | 1.09e-1 | 1.48e-1 | 1.22e-1 | 1.19e-1 | 46 | -6.08e-3 | +5.11e-3 | +7.91e-5 | -7.02e-5 |
| 76 | 3.00e-1 | 2 | 1.06e-1 | 1.10e-1 | 1.08e-1 | 1.10e-1 | 47 | -2.58e-3 | +8.46e-4 | -8.69e-4 | -2.05e-4 |
| 77 | 3.00e-1 | 1 | 1.10e-1 | 1.10e-1 | 1.10e-1 | 1.10e-1 | 287 | -1.59e-5 | -1.59e-5 | -1.59e-5 | -1.86e-4 |
| 78 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 281 | +2.68e-3 | +2.68e-3 | +2.68e-3 | +1.00e-4 |
| 79 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 322 | -2.02e-4 | -2.02e-4 | -2.02e-4 | +7.01e-5 |
| 80 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 371 | +1.60e-5 | +1.60e-5 | +1.60e-5 | +6.47e-5 |
| 81 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 333 | +1.11e-4 | +1.11e-4 | +1.11e-4 | +6.94e-5 |
| 83 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 394 | -8.80e-5 | -8.80e-5 | -8.80e-5 | +5.36e-5 |
| 84 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 312 | +1.46e-4 | +1.46e-4 | +1.46e-4 | +6.28e-5 |
| 85 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 270 | -3.02e-4 | -3.02e-4 | -3.02e-4 | +2.64e-5 |
| 86 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 281 | -1.46e-4 | -1.46e-4 | -1.46e-4 | +9.16e-6 |
| 87 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 282 | +5.00e-5 | +5.00e-5 | +5.00e-5 | +1.32e-5 |
| 88 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 293 | +9.54e-6 | +9.54e-6 | +9.54e-6 | +1.29e-5 |
| 89 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 357 | +6.85e-5 | +6.85e-5 | +6.85e-5 | +1.84e-5 |
| 90 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 309 | +1.62e-4 | +1.62e-4 | +1.62e-4 | +3.28e-5 |
| 92 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 384 | -1.18e-4 | -1.18e-4 | -1.18e-4 | +1.77e-5 |
| 93 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 316 | +1.87e-4 | +1.87e-4 | +1.87e-4 | +3.47e-5 |
| 94 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 319 | -2.18e-4 | -2.18e-4 | -2.18e-4 | +9.42e-6 |
| 95 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 310 | +8.14e-5 | +8.14e-5 | +8.14e-5 | +1.66e-5 |
| 96 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 280 | -4.51e-5 | -4.51e-5 | -4.51e-5 | +1.04e-5 |
| 97 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 285 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -3.85e-6 |
| 98 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 318 | -1.47e-6 | -1.47e-6 | -1.47e-6 | -3.62e-6 |
| 99 | 3.00e-2 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 330 | +1.41e-4 | +1.41e-4 | +1.41e-4 | +1.09e-5 |
| 101 | 3.00e-2 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 366 | +1.22e-5 | +1.22e-5 | +1.22e-5 | +1.10e-5 |
| 102 | 3.00e-2 | 1 | 1.25e-1 | 1.25e-1 | 1.25e-1 | 1.25e-1 | 354 | -1.55e-3 | -1.55e-3 | -1.55e-3 | -1.45e-4 |
| 103 | 3.00e-2 | 1 | 2.54e-2 | 2.54e-2 | 2.54e-2 | 2.54e-2 | 309 | -5.15e-3 | -5.15e-3 | -5.15e-3 | -6.45e-4 |
| 104 | 3.00e-2 | 1 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 316 | -5.94e-5 | -5.94e-5 | -5.94e-5 | -5.87e-4 |
| 105 | 3.00e-2 | 1 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 281 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -5.03e-4 |
| 106 | 3.00e-2 | 1 | 2.63e-2 | 2.63e-2 | 2.63e-2 | 2.63e-2 | 300 | -5.51e-5 | -5.51e-5 | -5.51e-5 | -4.59e-4 |
| 107 | 3.00e-2 | 1 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 269 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -3.95e-4 |
| 108 | 3.00e-2 | 1 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 274 | -8.48e-5 | -8.48e-5 | -8.48e-5 | -3.64e-4 |
| 109 | 3.00e-2 | 1 | 2.85e-2 | 2.85e-2 | 2.85e-2 | 2.85e-2 | 333 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -3.11e-4 |
| 110 | 3.00e-2 | 1 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 296 | +2.78e-4 | +2.78e-4 | +2.78e-4 | -2.52e-4 |
| 112 | 3.00e-2 | 1 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 369 | +3.81e-5 | +3.81e-5 | +3.81e-5 | -2.23e-4 |
| 113 | 3.00e-2 | 1 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 315 | +2.91e-4 | +2.91e-4 | +2.91e-4 | -1.72e-4 |
| 114 | 3.00e-2 | 1 | 3.29e-2 | 3.29e-2 | 3.29e-2 | 3.29e-2 | 323 | -1.40e-4 | -1.40e-4 | -1.40e-4 | -1.69e-4 |
| 115 | 3.00e-2 | 1 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 324 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -1.40e-4 |
| 116 | 3.00e-2 | 1 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 282 | +1.60e-4 | +1.60e-4 | +1.60e-4 | -1.10e-4 |
| 117 | 3.00e-2 | 1 | 3.51e-2 | 3.51e-2 | 3.51e-2 | 3.51e-2 | 323 | -5.75e-5 | -5.75e-5 | -5.75e-5 | -1.04e-4 |
| 118 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 332 | +1.92e-4 | +1.92e-4 | +1.92e-4 | -7.48e-5 |
| 120 | 3.00e-2 | 1 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 385 | +3.88e-5 | +3.88e-5 | +3.88e-5 | -6.34e-5 |
| 121 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 311 | +3.20e-4 | +3.20e-4 | +3.20e-4 | -2.51e-5 |
| 122 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 326 | -2.26e-4 | -2.26e-4 | -2.26e-4 | -4.52e-5 |
| 123 | 3.00e-2 | 1 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 280 | +2.43e-4 | +2.43e-4 | +2.43e-4 | -1.63e-5 |
| 124 | 3.00e-2 | 1 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 281 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -3.24e-5 |
| 125 | 3.00e-2 | 1 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 277 | +1.53e-5 | +1.53e-5 | +1.53e-5 | -2.76e-5 |
| 126 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 289 | +9.84e-5 | +9.84e-5 | +9.84e-5 | -1.50e-5 |
| 127 | 3.00e-2 | 1 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 290 | +8.52e-5 | +8.52e-5 | +8.52e-5 | -5.00e-6 |
| 128 | 3.00e-2 | 1 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 289 | +3.85e-5 | +3.85e-5 | +3.85e-5 | -6.53e-7 |
| 129 | 3.00e-2 | 1 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 337 | +1.40e-4 | +1.40e-4 | +1.40e-4 | +1.34e-5 |
| 131 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 354 | +1.73e-4 | +1.73e-4 | +1.73e-4 | +2.94e-5 |
| 132 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 336 | +1.50e-4 | +1.50e-4 | +1.50e-4 | +4.15e-5 |
| 133 | 3.00e-2 | 1 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 331 | -4.44e-5 | -4.44e-5 | -4.44e-5 | +3.29e-5 |
| 134 | 3.00e-2 | 1 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 318 | +7.60e-5 | +7.60e-5 | +7.60e-5 | +3.72e-5 |
| 135 | 3.00e-2 | 1 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 328 | +8.78e-6 | +8.78e-6 | +8.78e-6 | +3.43e-5 |
| 137 | 3.00e-2 | 2 | 5.20e-2 | 5.39e-2 | 5.30e-2 | 5.39e-2 | 271 | +8.60e-5 | +1.34e-4 | +1.10e-4 | +4.89e-5 |
| 139 | 3.00e-2 | 2 | 4.85e-2 | 5.36e-2 | 5.10e-2 | 5.36e-2 | 251 | -3.21e-4 | +4.04e-4 | +4.16e-5 | +5.12e-5 |
| 141 | 3.00e-2 | 1 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 352 | -1.19e-4 | -1.19e-4 | -1.19e-4 | +3.41e-5 |
| 142 | 3.00e-2 | 1 | 5.75e-2 | 5.75e-2 | 5.75e-2 | 5.75e-2 | 272 | +4.14e-4 | +4.14e-4 | +4.14e-4 | +7.21e-5 |
| 143 | 3.00e-2 | 1 | 5.22e-2 | 5.22e-2 | 5.22e-2 | 5.22e-2 | 282 | -3.43e-4 | -3.43e-4 | -3.43e-4 | +3.06e-5 |
| 144 | 3.00e-2 | 1 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 295 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +4.07e-5 |
| 145 | 3.00e-2 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 298 | -3.81e-5 | -3.81e-5 | -3.81e-5 | +3.28e-5 |
| 146 | 3.00e-2 | 1 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 297 | +1.69e-4 | +1.69e-4 | +1.69e-4 | +4.64e-5 |
| 147 | 3.00e-2 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 292 | -4.61e-6 | -4.61e-6 | -4.61e-6 | +4.13e-5 |
| 148 | 3.00e-2 | 1 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 310 | +1.12e-4 | +1.12e-4 | +1.12e-4 | +4.84e-5 |
| 150 | 3.00e-3 | 2 | 5.97e-2 | 6.33e-2 | 6.15e-2 | 6.33e-2 | 292 | +6.06e-5 | +2.03e-4 | +1.32e-4 | +6.49e-5 |
| 152 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 356 | -6.70e-3 | -6.70e-3 | -6.70e-3 | -6.11e-4 |
| 153 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 327 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -5.38e-4 |
| 154 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 322 | +1.39e-5 | +1.39e-5 | +1.39e-5 | -4.83e-4 |
| 155 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 337 | -6.41e-5 | -6.41e-5 | -6.41e-5 | -4.41e-4 |
| 156 | 3.00e-3 | 1 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 299 | +5.20e-6 | +5.20e-6 | +5.20e-6 | -3.97e-4 |
| 158 | 3.00e-3 | 1 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 370 | -3.57e-5 | -3.57e-5 | -3.57e-5 | -3.61e-4 |
| 159 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 305 | +3.07e-4 | +3.07e-4 | +3.07e-4 | -2.94e-4 |
| 160 | 3.00e-3 | 1 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 279 | -2.68e-4 | -2.68e-4 | -2.68e-4 | -2.91e-4 |
| 161 | 3.00e-3 | 1 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 311 | -6.48e-5 | -6.48e-5 | -6.48e-5 | -2.69e-4 |
| 162 | 3.00e-3 | 1 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 303 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -2.25e-4 |
| 163 | 3.00e-3 | 1 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 290 | +3.53e-7 | +3.53e-7 | +3.53e-7 | -2.02e-4 |
| 164 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 298 | -4.45e-5 | -4.45e-5 | -4.45e-5 | -1.86e-4 |
| 165 | 3.00e-3 | 1 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 327 | +4.88e-5 | +4.88e-5 | +4.88e-5 | -1.63e-4 |
| 167 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 394 | +1.38e-4 | +1.38e-4 | +1.38e-4 | -1.33e-4 |
| 168 | 3.00e-3 | 1 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 330 | +2.75e-4 | +2.75e-4 | +2.75e-4 | -9.21e-5 |
| 169 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 315 | -3.57e-4 | -3.57e-4 | -3.57e-4 | -1.19e-4 |
| 170 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 326 | -5.69e-5 | -5.69e-5 | -5.69e-5 | -1.12e-4 |
| 171 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 331 | +7.40e-5 | +7.40e-5 | +7.40e-5 | -9.38e-5 |
| 172 | 3.00e-3 | 1 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 344 | +7.30e-5 | +7.30e-5 | +7.30e-5 | -7.71e-5 |
| 174 | 3.00e-3 | 1 | 6.81e-3 | 6.81e-3 | 6.81e-3 | 6.81e-3 | 364 | +8.02e-5 | +8.02e-5 | +8.02e-5 | -6.14e-5 |
| 175 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 290 | -3.64e-5 | -3.64e-5 | -3.64e-5 | -5.89e-5 |
| 176 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 320 | -1.03e-4 | -1.03e-4 | -1.03e-4 | -6.33e-5 |
| 177 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 312 | +4.15e-5 | +4.15e-5 | +4.15e-5 | -5.28e-5 |
| 178 | 3.00e-3 | 1 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 326 | +6.30e-5 | +6.30e-5 | +6.30e-5 | -4.12e-5 |
| 179 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 310 | -9.93e-6 | -9.93e-6 | -9.93e-6 | -3.81e-5 |
| 180 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 290 | +4.36e-6 | +4.36e-6 | +4.36e-6 | -3.38e-5 |
| 182 | 3.00e-3 | 1 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 371 | -2.14e-5 | -2.14e-5 | -2.14e-5 | -3.26e-5 |
| 183 | 3.00e-3 | 1 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 290 | +3.02e-4 | +3.02e-4 | +3.02e-4 | +8.42e-7 |
| 184 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 320 | -3.06e-4 | -3.06e-4 | -3.06e-4 | -2.99e-5 |
| 185 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 343 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -1.33e-5 |
| 186 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 305 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +1.24e-6 |
| 187 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 286 | -2.41e-4 | -2.41e-4 | -2.41e-4 | -2.29e-5 |
| 188 | 3.00e-3 | 1 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 288 | +3.55e-5 | +3.55e-5 | +3.55e-5 | -1.71e-5 |
| 189 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 320 | -1.39e-5 | -1.39e-5 | -1.39e-5 | -1.68e-5 |
| 191 | 3.00e-3 | 2 | 6.99e-3 | 7.60e-3 | 7.30e-3 | 7.60e-3 | 270 | +8.62e-5 | +3.07e-4 | +1.97e-4 | +2.49e-5 |
| 193 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 353 | -4.07e-4 | -4.07e-4 | -4.07e-4 | -1.83e-5 |
| 194 | 3.00e-3 | 1 | 7.44e-3 | 7.44e-3 | 7.44e-3 | 7.44e-3 | 287 | +4.28e-4 | +4.28e-4 | +4.28e-4 | +2.64e-5 |
| 195 | 3.00e-3 | 1 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 302 | -2.98e-4 | -2.98e-4 | -2.98e-4 | -6.08e-6 |
| 196 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 303 | -3.41e-5 | -3.41e-5 | -3.41e-5 | -8.88e-6 |
| 197 | 3.00e-3 | 1 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 300 | +1.67e-4 | +1.67e-4 | +1.67e-4 | +8.74e-6 |
| 198 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 333 | -8.76e-5 | -8.76e-5 | -8.76e-5 | -8.89e-7 |

