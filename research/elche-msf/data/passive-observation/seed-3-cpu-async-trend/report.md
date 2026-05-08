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
| cpu-async | 0.063964 | 0.9168 | +0.0043 | 1917.3 | 646 | 83.6 | 100% | 100% | 11.8 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9168 | cpu-async | - | - |

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
| cpu-async | 2.0257 | 0.7587 | 0.5817 | 0.5270 | 0.4914 | 0.4719 | 0.4534 | 0.4434 | 0.4732 | 0.4734 | 0.2140 | 0.1783 | 0.1603 | 0.1452 | 0.1445 | 0.0832 | 0.0765 | 0.0717 | 0.0659 | 0.0640 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4049 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3013 | 3.4 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2937 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 392 | 390 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 1771.3 | 1.5 | epoch-boundary(184) |
| cpu-async | gpu1 | 1771.4 | 1.4 | epoch-boundary(184) |
| cpu-async | gpu1 | 1026.5 | 1.1 | epoch-boundary(106) |
| cpu-async | gpu2 | 1026.4 | 1.1 | epoch-boundary(106) |
| cpu-async | gpu1 | 1810.6 | 0.9 | epoch-boundary(188) |
| cpu-async | gpu2 | 1810.6 | 0.8 | epoch-boundary(188) |
| cpu-async | gpu1 | 1304.3 | 0.7 | epoch-boundary(135) |
| cpu-async | gpu2 | 950.0 | 0.6 | epoch-boundary(98) |
| cpu-async | gpu2 | 1189.2 | 0.5 | epoch-boundary(123) |
| cpu-async | gpu1 | 1189.1 | 0.5 | epoch-boundary(123) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.5s | 0.0s | 0.5s |
| resnet-graph | cpu-async | gpu1 | 5.1s | 0.0s | 0.0s | 0.0s | 5.7s |
| resnet-graph | cpu-async | gpu2 | 5.0s | 0.0s | 0.0s | 0.0s | 5.6s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 336 | 0 | 646 | 83.6 | 1526/9413 | 646 | 83.6 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 188.0 | 9.8% |

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
| resnet-graph | cpu-async | 196 | 646 | 0 | 7.53e-3 | -1.94e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 646 | 1.02e-1 | 5.39e-2 | 3.88e-3 | 5.11e-1 | 29.9 | -1.79e-4 | 1.30e-3 |
| resnet-graph | cpu-async | 1 | 646 | 1.02e-1 | 5.35e-2 | 3.55e-3 | 4.44e-1 | 33.7 | -1.95e-4 | 1.26e-3 |
| resnet-graph | cpu-async | 2 | 646 | 1.02e-1 | 5.30e-2 | 3.57e-3 | 4.45e-1 | 36.4 | -1.95e-4 | 1.24e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9840 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9858 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9873 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 83 (1,2,4,5,6,7,8,9…148,149) | 0 (—) | — | 1,2,4,5,6,7,8,9…148,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 10 | 10 |
| resnet-graph | cpu-async | 0e0 | 5 | 5 | 5 |
| resnet-graph | cpu-async | 0e0 | 10 | 2 | 2 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 498 | +0.100 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 56 | +0.079 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 88 | -0.098 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 644 | -0.050 | 195 | +0.487 | +0.641 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 645 | 3.40e1–8.07e1 | 6.85e1 | 2.12e-3 | 3.54e-3 | 5.08e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 500 | 32–78024 | +3.775e-6 | 0.124 | +3.984e-6 | 0.156 | 100 | +8.781e-6 | 0.577 | 32–891 | +9.901e-4 | 0.540 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 489 | 878–78024 | +4.931e-6 | 0.267 | +5.051e-6 | 0.308 | 99 | +9.308e-6 | 0.639 | 74–891 | +1.053e-3 | 0.786 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 57 | 78759–116124 | +9.286e-6 | 0.147 | +9.252e-6 | 0.152 | 49 | +8.896e-6 | 0.127 | 476–833 | -1.093e-3 | 0.142 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 89 | 116684–155821 | -9.204e-6 | 0.044 | -9.084e-6 | 0.043 | 47 | -4.829e-6 | 0.015 | 196–992 | +1.147e-3 | 0.248 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.901e-4 | r0: +9.658e-4, r1: +1.005e-3, r2: +1.001e-3 | r0: 0.516, r1: 0.516, r2: 0.533 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.053e-3 | r0: +1.029e-3, r1: +1.067e-3, r2: +1.062e-3 | r0: 0.772, r1: 0.738, r2: 0.750 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | -1.093e-3 | r0: -1.145e-3, r1: -1.106e-3, r2: -1.029e-3 | r0: 0.151, r1: 0.150, r2: 0.124 | 1.11× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | +1.147e-3 | r0: +1.129e-3, r1: +1.163e-3, r2: +1.149e-3 | r0: 0.243, r1: 0.252, r2: 0.247 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇██████▆▄▄▅▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▂▂▂` | `▁▆▇▆▆▆▇▇▆▇▆▇▆▇▇▇▇▇▇▇█████▇▇▇█████████▇▆▇▇████▇███` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.23e-1 | 5.11e-1 | 2.36e-1 | 1.48e-1 | 35 | -3.20e-2 | +1.08e-2 | -9.32e-3 | -6.90e-3 |
| 1 | 3.00e-1 | 11 | 9.05e-2 | 1.91e-1 | 1.09e-1 | 1.08e-1 | 36 | -1.44e-2 | +3.26e-3 | -1.31e-3 | -2.06e-3 |
| 2 | 3.00e-1 | 4 | 1.01e-1 | 1.41e-1 | 1.13e-1 | 1.01e-1 | 34 | -7.56e-3 | +3.65e-3 | -1.39e-3 | -1.83e-3 |
| 3 | 3.00e-1 | 8 | 1.01e-1 | 1.51e-1 | 1.13e-1 | 1.11e-1 | 35 | -8.07e-3 | +5.12e-3 | -6.48e-4 | -1.02e-3 |
| 4 | 3.00e-1 | 8 | 1.03e-1 | 1.55e-1 | 1.16e-1 | 1.14e-1 | 34 | -8.29e-3 | +4.51e-3 | -5.46e-4 | -6.39e-4 |
| 5 | 3.00e-1 | 8 | 1.00e-1 | 1.54e-1 | 1.12e-1 | 1.11e-1 | 35 | -1.17e-2 | +4.01e-3 | -9.83e-4 | -6.49e-4 |
| 6 | 3.00e-1 | 8 | 1.09e-1 | 1.60e-1 | 1.19e-1 | 1.13e-1 | 37 | -8.72e-3 | +4.22e-3 | -7.70e-4 | -6.36e-4 |
| 7 | 3.00e-1 | 4 | 1.20e-1 | 1.59e-1 | 1.33e-1 | 1.20e-1 | 44 | -4.50e-3 | +3.90e-3 | -5.19e-4 | -6.40e-4 |
| 8 | 3.00e-1 | 6 | 1.25e-1 | 1.54e-1 | 1.32e-1 | 1.30e-1 | 47 | -4.42e-3 | +3.06e-3 | -1.35e-4 | -3.91e-4 |
| 9 | 3.00e-1 | 5 | 1.17e-1 | 1.55e-1 | 1.29e-1 | 1.17e-1 | 44 | -4.46e-3 | +2.18e-3 | -8.00e-4 | -5.78e-4 |
| 10 | 3.00e-1 | 7 | 1.07e-1 | 1.53e-1 | 1.20e-1 | 1.07e-1 | 34 | -6.14e-3 | +3.17e-3 | -8.87e-4 | -7.51e-4 |
| 11 | 3.00e-1 | 7 | 1.00e-1 | 1.52e-1 | 1.12e-1 | 1.08e-1 | 34 | -8.23e-3 | +4.30e-3 | -8.45e-4 | -7.01e-4 |
| 12 | 3.00e-1 | 8 | 9.55e-2 | 1.52e-1 | 1.10e-1 | 1.11e-1 | 36 | -1.29e-2 | +4.13e-3 | -1.09e-3 | -7.00e-4 |
| 13 | 3.00e-1 | 7 | 1.09e-1 | 1.52e-1 | 1.20e-1 | 1.15e-1 | 35 | -6.34e-3 | +3.88e-3 | -5.82e-4 | -6.01e-4 |
| 14 | 3.00e-1 | 6 | 1.06e-1 | 1.52e-1 | 1.17e-1 | 1.13e-1 | 37 | -9.29e-3 | +3.60e-3 | -9.44e-4 | -6.91e-4 |
| 15 | 3.00e-1 | 10 | 1.03e-1 | 1.54e-1 | 1.14e-1 | 1.11e-1 | 37 | -6.23e-3 | +3.68e-3 | -4.76e-4 | -4.44e-4 |
| 16 | 3.00e-1 | 5 | 1.03e-1 | 1.44e-1 | 1.14e-1 | 1.03e-1 | 32 | -8.06e-3 | +3.61e-3 | -1.37e-3 | -8.29e-4 |
| 17 | 3.00e-1 | 7 | 1.03e-1 | 1.46e-1 | 1.12e-1 | 1.03e-1 | 32 | -8.35e-3 | +4.65e-3 | -8.77e-4 | -8.23e-4 |
| 18 | 3.00e-1 | 10 | 9.85e-2 | 1.54e-1 | 1.13e-1 | 1.10e-1 | 35 | -1.16e-2 | +5.41e-3 | -7.18e-4 | -6.16e-4 |
| 19 | 3.00e-1 | 5 | 1.06e-1 | 1.47e-1 | 1.17e-1 | 1.12e-1 | 35 | -8.45e-3 | +3.57e-3 | -9.36e-4 | -7.05e-4 |
| 20 | 3.00e-1 | 8 | 1.00e-1 | 1.50e-1 | 1.11e-1 | 1.10e-1 | 35 | -9.58e-3 | +4.09e-3 | -7.77e-4 | -5.82e-4 |
| 21 | 3.00e-1 | 10 | 1.01e-1 | 1.49e-1 | 1.13e-1 | 1.14e-1 | 36 | -7.68e-3 | +4.22e-3 | -4.28e-4 | -3.53e-4 |
| 22 | 3.00e-1 | 4 | 1.03e-1 | 1.58e-1 | 1.22e-1 | 1.08e-1 | 35 | -7.47e-3 | +4.07e-3 | -1.67e-3 | -8.23e-4 |
| 23 | 3.00e-1 | 8 | 1.09e-1 | 1.48e-1 | 1.20e-1 | 1.09e-1 | 35 | -6.38e-3 | +3.95e-3 | -5.37e-4 | -6.93e-4 |
| 24 | 3.00e-1 | 6 | 1.02e-1 | 1.57e-1 | 1.17e-1 | 1.24e-1 | 54 | -1.07e-2 | +4.38e-3 | -9.98e-4 | -6.94e-4 |
| 25 | 3.00e-1 | 4 | 1.31e-1 | 1.62e-1 | 1.40e-1 | 1.32e-1 | 56 | -2.91e-3 | +2.89e-3 | -1.87e-4 | -5.44e-4 |
| 26 | 3.00e-1 | 7 | 1.16e-1 | 1.54e-1 | 1.26e-1 | 1.28e-1 | 49 | -5.64e-3 | +1.93e-3 | -3.90e-4 | -3.84e-4 |
| 27 | 3.00e-1 | 5 | 1.15e-1 | 1.51e-1 | 1.26e-1 | 1.19e-1 | 44 | -3.97e-3 | +2.19e-3 | -6.50e-4 | -4.82e-4 |
| 28 | 3.00e-1 | 6 | 1.21e-1 | 1.52e-1 | 1.28e-1 | 1.23e-1 | 46 | -3.76e-3 | +3.16e-3 | -2.76e-4 | -3.89e-4 |
| 29 | 3.00e-1 | 6 | 1.07e-1 | 1.50e-1 | 1.23e-1 | 1.32e-1 | 56 | -7.94e-3 | +2.98e-3 | -5.05e-4 | -3.34e-4 |
| 30 | 3.00e-1 | 5 | 1.29e-1 | 1.63e-1 | 1.39e-1 | 1.29e-1 | 56 | -2.82e-3 | +2.12e-3 | -4.07e-4 | -3.77e-4 |
| 31 | 3.00e-1 | 5 | 1.19e-1 | 1.60e-1 | 1.33e-1 | 1.22e-1 | 48 | -2.67e-3 | +2.36e-3 | -5.35e-4 | -4.53e-4 |
| 32 | 3.00e-1 | 6 | 1.09e-1 | 1.49e-1 | 1.22e-1 | 1.12e-1 | 38 | -4.04e-3 | +2.67e-3 | -6.77e-4 | -5.62e-4 |
| 33 | 3.00e-1 | 6 | 1.05e-1 | 1.59e-1 | 1.19e-1 | 1.06e-1 | 37 | -5.10e-3 | +3.80e-3 | -1.00e-3 | -7.61e-4 |
| 34 | 3.00e-1 | 7 | 1.15e-1 | 1.47e-1 | 1.21e-1 | 1.15e-1 | 40 | -5.30e-3 | +4.52e-3 | -2.21e-4 | -4.89e-4 |
| 35 | 3.00e-1 | 6 | 1.11e-1 | 1.56e-1 | 1.20e-1 | 1.11e-1 | 38 | -7.93e-3 | +3.71e-3 | -9.61e-4 | -6.81e-4 |
| 36 | 3.00e-1 | 8 | 1.10e-1 | 1.53e-1 | 1.22e-1 | 1.22e-1 | 44 | -7.69e-3 | +4.05e-3 | -3.94e-4 | -4.36e-4 |
| 37 | 3.00e-1 | 5 | 1.15e-1 | 1.59e-1 | 1.26e-1 | 1.15e-1 | 41 | -5.84e-3 | +3.06e-3 | -9.13e-4 | -6.31e-4 |
| 38 | 3.00e-1 | 5 | 1.20e-1 | 1.55e-1 | 1.34e-1 | 1.41e-1 | 60 | -5.07e-3 | +3.52e-3 | +2.84e-5 | -3.29e-4 |
| 39 | 3.00e-1 | 6 | 1.15e-1 | 1.66e-1 | 1.29e-1 | 1.16e-1 | 41 | -4.40e-3 | +1.67e-3 | -1.03e-3 | -6.32e-4 |
| 40 | 3.00e-1 | 5 | 1.17e-1 | 1.62e-1 | 1.28e-1 | 1.20e-1 | 44 | -6.55e-3 | +3.67e-3 | -7.34e-4 | -6.65e-4 |
| 41 | 3.00e-1 | 7 | 1.12e-1 | 1.57e-1 | 1.24e-1 | 1.28e-1 | 49 | -9.77e-3 | +3.04e-3 | -5.88e-4 | -5.08e-4 |
| 42 | 3.00e-1 | 5 | 1.14e-1 | 1.58e-1 | 1.28e-1 | 1.14e-1 | 41 | -4.29e-3 | +2.57e-3 | -9.54e-4 | -7.07e-4 |
| 43 | 3.00e-1 | 6 | 1.12e-1 | 1.58e-1 | 1.23e-1 | 1.15e-1 | 38 | -7.37e-3 | +3.75e-3 | -7.80e-4 | -7.14e-4 |
| 44 | 3.00e-1 | 6 | 1.17e-1 | 1.43e-1 | 1.23e-1 | 1.19e-1 | 41 | -4.01e-3 | +3.25e-3 | -1.91e-4 | -4.71e-4 |
| 45 | 3.00e-1 | 7 | 1.13e-1 | 1.56e-1 | 1.33e-1 | 1.41e-1 | 58 | -6.72e-3 | +3.28e-3 | -2.16e-4 | -2.64e-4 |
| 46 | 3.00e-1 | 3 | 1.28e-1 | 1.76e-1 | 1.48e-1 | 1.28e-1 | 48 | -4.00e-3 | +1.99e-3 | -1.30e-3 | -5.80e-4 |
| 47 | 3.00e-1 | 5 | 1.14e-1 | 1.58e-1 | 1.28e-1 | 1.14e-1 | 41 | -4.71e-3 | +2.47e-3 | -1.00e-3 | -7.67e-4 |
| 48 | 3.00e-1 | 6 | 1.14e-1 | 1.59e-1 | 1.29e-1 | 1.24e-1 | 48 | -5.38e-3 | +3.49e-3 | -3.34e-4 | -5.84e-4 |
| 49 | 3.00e-1 | 5 | 1.14e-1 | 1.69e-1 | 1.31e-1 | 1.14e-1 | 42 | -5.45e-3 | +2.99e-3 | -1.14e-3 | -8.25e-4 |
| 50 | 3.00e-1 | 8 | 1.11e-1 | 1.59e-1 | 1.24e-1 | 1.24e-1 | 45 | -5.99e-3 | +4.07e-3 | -2.59e-4 | -4.39e-4 |
| 51 | 3.00e-1 | 4 | 1.16e-1 | 1.59e-1 | 1.30e-1 | 1.16e-1 | 41 | -5.80e-3 | +2.84e-3 | -1.24e-3 | -7.38e-4 |
| 52 | 3.00e-1 | 6 | 1.17e-1 | 1.57e-1 | 1.26e-1 | 1.18e-1 | 41 | -6.30e-3 | +3.64e-3 | -5.75e-4 | -6.53e-4 |
| 53 | 3.00e-1 | 6 | 1.13e-1 | 1.51e-1 | 1.23e-1 | 1.17e-1 | 43 | -5.09e-3 | +3.22e-3 | -5.42e-4 | -5.87e-4 |
| 54 | 3.00e-1 | 7 | 1.12e-1 | 1.57e-1 | 1.23e-1 | 1.13e-1 | 37 | -4.38e-3 | +3.54e-3 | -5.23e-4 | -5.53e-4 |
| 55 | 3.00e-1 | 6 | 1.15e-1 | 1.45e-1 | 1.26e-1 | 1.15e-1 | 37 | -3.47e-3 | +3.36e-3 | -4.54e-4 | -5.75e-4 |
| 56 | 3.00e-1 | 8 | 1.04e-1 | 1.53e-1 | 1.17e-1 | 1.21e-1 | 47 | -1.03e-2 | +3.50e-3 | -7.18e-4 | -4.77e-4 |
| 57 | 3.00e-1 | 4 | 1.17e-1 | 1.68e-1 | 1.35e-1 | 1.17e-1 | 42 | -4.28e-3 | +3.64e-3 | -9.59e-4 | -6.90e-4 |
| 58 | 3.00e-1 | 7 | 1.09e-1 | 1.53e-1 | 1.19e-1 | 1.15e-1 | 38 | -7.91e-3 | +3.31e-3 | -7.37e-4 | -6.46e-4 |
| 59 | 3.00e-1 | 5 | 1.21e-1 | 1.54e-1 | 1.37e-1 | 1.38e-1 | 61 | -6.27e-3 | +3.65e-3 | -9.58e-5 | -4.20e-4 |
| 60 | 3.00e-1 | 6 | 1.20e-1 | 1.66e-1 | 1.39e-1 | 1.20e-1 | 47 | -2.16e-3 | +2.06e-3 | -6.78e-4 | -5.92e-4 |
| 61 | 3.00e-1 | 3 | 1.33e-1 | 1.56e-1 | 1.42e-1 | 1.38e-1 | 58 | -3.44e-3 | +3.14e-3 | +1.23e-4 | -4.20e-4 |
| 62 | 3.00e-1 | 5 | 1.16e-1 | 1.59e-1 | 1.29e-1 | 1.16e-1 | 42 | -3.50e-3 | +1.67e-3 | -1.00e-3 | -6.62e-4 |
| 63 | 3.00e-1 | 7 | 1.16e-1 | 1.57e-1 | 1.26e-1 | 1.21e-1 | 49 | -4.68e-3 | +3.41e-3 | -3.92e-4 | -5.00e-4 |
| 64 | 3.00e-1 | 4 | 1.28e-1 | 1.62e-1 | 1.40e-1 | 1.28e-1 | 55 | -3.09e-3 | +3.52e-3 | -2.53e-4 | -4.55e-4 |
| 65 | 3.00e-1 | 5 | 1.26e-1 | 1.63e-1 | 1.36e-1 | 1.26e-1 | 48 | -3.34e-3 | +2.52e-3 | -4.41e-4 | -4.65e-4 |
| 66 | 3.00e-1 | 8 | 1.07e-1 | 1.57e-1 | 1.20e-1 | 1.07e-1 | 36 | -5.75e-3 | +2.80e-3 | -8.61e-4 | -6.84e-4 |
| 67 | 3.00e-1 | 6 | 1.08e-1 | 1.55e-1 | 1.19e-1 | 1.13e-1 | 44 | -7.49e-3 | +4.82e-3 | -7.17e-4 | -6.66e-4 |
| 68 | 3.00e-1 | 5 | 1.22e-1 | 1.53e-1 | 1.31e-1 | 1.22e-1 | 45 | -4.68e-3 | +3.89e-3 | -2.79e-4 | -5.47e-4 |
| 69 | 3.00e-1 | 6 | 1.17e-1 | 1.57e-1 | 1.27e-1 | 1.22e-1 | 44 | -4.42e-3 | +3.10e-3 | -4.73e-4 | -4.87e-4 |
| 70 | 3.00e-1 | 6 | 1.16e-1 | 1.58e-1 | 1.27e-1 | 1.25e-1 | 47 | -5.48e-3 | +3.25e-3 | -3.73e-4 | -3.96e-4 |
| 71 | 3.00e-1 | 6 | 1.15e-1 | 1.65e-1 | 1.27e-1 | 1.25e-1 | 51 | -8.20e-3 | +3.16e-3 | -8.23e-4 | -5.27e-4 |
| 72 | 3.00e-1 | 7 | 1.15e-1 | 1.65e-1 | 1.26e-1 | 1.21e-1 | 46 | -7.22e-3 | +2.99e-3 | -7.54e-4 | -5.71e-4 |
| 73 | 3.00e-1 | 4 | 1.29e-1 | 1.52e-1 | 1.37e-1 | 1.34e-1 | 50 | -3.29e-3 | +2.93e-3 | +8.45e-5 | -3.62e-4 |
| 74 | 3.00e-1 | 5 | 1.10e-1 | 1.64e-1 | 1.33e-1 | 1.42e-1 | 59 | -9.16e-3 | +3.50e-3 | -8.66e-4 | -4.77e-4 |
| 75 | 3.00e-1 | 4 | 1.40e-1 | 1.69e-1 | 1.48e-1 | 1.40e-1 | 62 | -2.83e-3 | +1.86e-3 | -2.92e-4 | -4.28e-4 |
| 76 | 3.00e-1 | 6 | 1.31e-1 | 1.70e-1 | 1.41e-1 | 1.36e-1 | 54 | -2.61e-3 | +1.94e-3 | -2.66e-4 | -3.40e-4 |
| 77 | 3.00e-1 | 1 | 1.29e-1 | 1.29e-1 | 1.29e-1 | 1.29e-1 | 52 | -9.30e-4 | -9.30e-4 | -9.30e-4 | -3.99e-4 |
| 78 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 284 | +1.89e-3 | +1.89e-3 | +1.89e-3 | -1.70e-4 |
| 79 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 254 | +6.74e-6 | +6.74e-6 | +6.74e-6 | -1.52e-4 |
| 80 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 263 | +3.05e-5 | +3.05e-5 | +3.05e-5 | -1.34e-4 |
| 81 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 267 | -3.00e-5 | -3.00e-5 | -3.00e-5 | -1.24e-4 |
| 82 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 295 | +7.63e-5 | +7.63e-5 | +7.63e-5 | -1.04e-4 |
| 83 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 302 | +5.26e-5 | +5.26e-5 | +5.26e-5 | -8.80e-5 |
| 84 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 324 | +1.14e-5 | +1.14e-5 | +1.14e-5 | -7.81e-5 |
| 85 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 291 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -8.18e-5 |
| 86 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 263 | -8.78e-5 | -8.78e-5 | -8.78e-5 | -8.24e-5 |
| 87 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 298 | +1.14e-4 | +1.14e-4 | +1.14e-4 | -6.27e-5 |
| 88 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 279 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -6.85e-5 |
| 89 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 279 | +1.41e-5 | +1.41e-5 | +1.41e-5 | -6.02e-5 |
| 90 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 293 | +6.17e-5 | +6.17e-5 | +6.17e-5 | -4.80e-5 |
| 91 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 268 | -8.23e-5 | -8.23e-5 | -8.23e-5 | -5.15e-5 |
| 92 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 282 | +3.51e-5 | +3.51e-5 | +3.51e-5 | -4.28e-5 |
| 93 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 298 | +7.39e-5 | +7.39e-5 | +7.39e-5 | -3.11e-5 |
| 94 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 262 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -4.13e-5 |
| 95 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 307 | +1.76e-4 | +1.76e-4 | +1.76e-4 | -1.96e-5 |
| 96 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 279 | -1.20e-4 | -1.20e-4 | -1.20e-4 | -2.97e-5 |
| 97 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 278 | -3.20e-5 | -3.20e-5 | -3.20e-5 | -2.99e-5 |
| 98 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 260 | -5.57e-5 | -5.57e-5 | -5.57e-5 | -3.25e-5 |
| 99 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 282 | +7.11e-5 | +7.11e-5 | +7.11e-5 | -2.21e-5 |
| 100 | 3.00e-2 | 1 | 1.61e-1 | 1.61e-1 | 1.61e-1 | 1.61e-1 | 271 | -1.19e-3 | -1.19e-3 | -1.19e-3 | -1.39e-4 |
| 101 | 3.00e-2 | 1 | 7.78e-2 | 7.78e-2 | 7.78e-2 | 7.78e-2 | 294 | -2.47e-3 | -2.47e-3 | -2.47e-3 | -3.72e-4 |
| 102 | 3.00e-2 | 1 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 259 | -2.20e-3 | -2.20e-3 | -2.20e-3 | -5.55e-4 |
| 103 | 3.00e-2 | 1 | 3.18e-2 | 3.18e-2 | 3.18e-2 | 3.18e-2 | 277 | -1.17e-3 | -1.17e-3 | -1.17e-3 | -6.17e-4 |
| 104 | 3.00e-2 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 258 | -4.04e-4 | -4.04e-4 | -4.04e-4 | -5.95e-4 |
| 105 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 307 | +2.13e-4 | +2.13e-4 | +2.13e-4 | -5.15e-4 |
| 106 | 3.00e-2 | 1 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 254 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -4.77e-4 |
| 107 | 3.00e-2 | 1 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 291 | +1.85e-4 | +1.85e-4 | +1.85e-4 | -4.10e-4 |
| 108 | 3.00e-2 | 1 | 3.16e-2 | 3.16e-2 | 3.16e-2 | 3.16e-2 | 273 | +4.45e-5 | +4.45e-5 | +4.45e-5 | -3.65e-4 |
| 109 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 278 | +1.89e-5 | +1.89e-5 | +1.89e-5 | -3.27e-4 |
| 110 | 3.00e-2 | 1 | 3.25e-2 | 3.25e-2 | 3.25e-2 | 3.25e-2 | 263 | +9.03e-5 | +9.03e-5 | +9.03e-5 | -2.85e-4 |
| 111 | 3.00e-2 | 1 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 279 | +1.15e-4 | +1.15e-4 | +1.15e-4 | -2.45e-4 |
| 112 | 3.00e-2 | 1 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 251 | -1.64e-5 | -1.64e-5 | -1.64e-5 | -2.22e-4 |
| 113 | 3.00e-2 | 1 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 277 | +1.60e-4 | +1.60e-4 | +1.60e-4 | -1.84e-4 |
| 114 | 3.00e-2 | 1 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 240 | -3.20e-5 | -3.20e-5 | -3.20e-5 | -1.69e-4 |
| 115 | 3.00e-2 | 1 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 255 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -1.39e-4 |
| 116 | 3.00e-2 | 1 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 255 | +7.88e-5 | +7.88e-5 | +7.88e-5 | -1.18e-4 |
| 117 | 3.00e-2 | 1 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 280 | +1.38e-4 | +1.38e-4 | +1.38e-4 | -9.20e-5 |
| 118 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 283 | +1.49e-4 | +1.49e-4 | +1.49e-4 | -6.79e-5 |
| 119 | 3.00e-2 | 1 | 3.83e-2 | 3.83e-2 | 3.83e-2 | 3.83e-2 | 259 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -7.34e-5 |
| 120 | 3.00e-2 | 1 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 284 | +1.91e-4 | +1.91e-4 | +1.91e-4 | -4.70e-5 |
| 121 | 3.00e-2 | 1 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 264 | -5.80e-5 | -5.80e-5 | -5.80e-5 | -4.81e-5 |
| 122 | 3.00e-2 | 1 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 233 | +2.29e-5 | +2.29e-5 | +2.29e-5 | -4.10e-5 |
| 123 | 3.00e-2 | 2 | 3.96e-2 | 4.19e-2 | 4.08e-2 | 3.96e-2 | 211 | -2.63e-4 | +1.78e-4 | -4.28e-5 | -4.35e-5 |
| 124 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 236 | +1.83e-4 | +1.83e-4 | +1.83e-4 | -2.09e-5 |
| 125 | 3.00e-2 | 1 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 284 | +3.83e-4 | +3.83e-4 | +3.83e-4 | +1.95e-5 |
| 126 | 3.00e-2 | 1 | 4.41e-2 | 4.41e-2 | 4.41e-2 | 4.41e-2 | 246 | -1.85e-4 | -1.85e-4 | -1.85e-4 | -9.31e-7 |
| 127 | 3.00e-2 | 2 | 4.11e-2 | 4.15e-2 | 4.13e-2 | 4.11e-2 | 187 | -2.84e-4 | -5.01e-5 | -1.67e-4 | -3.14e-5 |
| 128 | 3.00e-2 | 1 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 187 | -8.75e-5 | -8.75e-5 | -8.75e-5 | -3.70e-5 |
| 129 | 3.00e-2 | 1 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 216 | +3.81e-4 | +3.81e-4 | +3.81e-4 | +4.78e-6 |
| 130 | 3.00e-2 | 2 | 4.51e-2 | 4.56e-2 | 4.53e-2 | 4.56e-2 | 208 | +5.85e-5 | +1.09e-4 | +8.38e-5 | +1.95e-5 |
| 131 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 245 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +3.28e-5 |
| 132 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 224 | -1.81e-6 | -1.81e-6 | -1.81e-6 | +2.93e-5 |
| 133 | 3.00e-2 | 1 | 4.76e-2 | 4.76e-2 | 4.76e-2 | 4.76e-2 | 234 | +2.13e-5 | +2.13e-5 | +2.13e-5 | +2.85e-5 |
| 134 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 236 | +7.79e-5 | +7.79e-5 | +7.79e-5 | +3.35e-5 |
| 135 | 3.00e-2 | 2 | 4.68e-2 | 4.69e-2 | 4.68e-2 | 4.68e-2 | 196 | -1.66e-4 | -1.02e-5 | -8.80e-5 | +1.12e-5 |
| 136 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 219 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +2.53e-5 |
| 137 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 223 | +2.15e-4 | +2.15e-4 | +2.15e-4 | +4.43e-5 |
| 138 | 3.00e-2 | 2 | 4.90e-2 | 5.00e-2 | 4.95e-2 | 4.90e-2 | 196 | -1.03e-4 | -6.76e-5 | -8.51e-5 | +1.95e-5 |
| 139 | 3.00e-2 | 1 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 215 | +2.17e-4 | +2.17e-4 | +2.17e-4 | +3.93e-5 |
| 140 | 3.00e-2 | 1 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 235 | +1.18e-4 | +1.18e-4 | +1.18e-4 | +4.72e-5 |
| 141 | 3.00e-2 | 1 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 220 | -1.22e-4 | -1.22e-4 | -1.22e-4 | +3.03e-5 |
| 142 | 3.00e-2 | 2 | 5.01e-2 | 5.15e-2 | 5.08e-2 | 5.01e-2 | 185 | -1.52e-4 | +1.25e-5 | -6.96e-5 | +1.05e-5 |
| 143 | 3.00e-2 | 1 | 5.25e-2 | 5.25e-2 | 5.25e-2 | 5.25e-2 | 202 | +2.37e-4 | +2.37e-4 | +2.37e-4 | +3.31e-5 |
| 144 | 3.00e-2 | 2 | 5.11e-2 | 5.24e-2 | 5.17e-2 | 5.11e-2 | 185 | -1.31e-4 | -1.56e-5 | -7.34e-5 | +1.23e-5 |
| 145 | 3.00e-2 | 1 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 207 | +2.19e-4 | +2.19e-4 | +2.19e-4 | +3.29e-5 |
| 146 | 3.00e-2 | 2 | 5.38e-2 | 5.41e-2 | 5.40e-2 | 5.38e-2 | 194 | -2.81e-5 | +5.84e-5 | +1.51e-5 | +2.91e-5 |
| 147 | 3.00e-2 | 1 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 211 | +2.57e-5 | +2.57e-5 | +2.57e-5 | +2.87e-5 |
| 148 | 3.00e-2 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 219 | +1.91e-4 | +1.91e-4 | +1.91e-4 | +4.50e-5 |
| 149 | 3.00e-3 | 2 | 5.37e-2 | 5.65e-2 | 5.51e-2 | 5.37e-2 | 165 | -3.08e-4 | +1.01e-5 | -1.49e-4 | +6.54e-6 |
| 150 | 3.00e-3 | 1 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 212 | -3.12e-3 | -3.12e-3 | -3.12e-3 | -3.06e-4 |
| 151 | 3.00e-3 | 2 | 8.28e-3 | 1.42e-2 | 1.12e-2 | 8.28e-3 | 155 | -3.48e-3 | -3.47e-3 | -3.47e-3 | -9.08e-4 |
| 152 | 3.00e-3 | 1 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 172 | -1.78e-3 | -1.78e-3 | -1.78e-3 | -9.95e-4 |
| 153 | 3.00e-3 | 2 | 5.31e-3 | 5.58e-3 | 5.45e-3 | 5.31e-3 | 168 | -4.84e-4 | -2.95e-4 | -3.89e-4 | -8.79e-4 |
| 154 | 3.00e-3 | 2 | 5.03e-3 | 5.66e-3 | 5.35e-3 | 5.03e-3 | 162 | -7.35e-4 | +3.14e-4 | -2.11e-4 | -7.57e-4 |
| 155 | 3.00e-3 | 1 | 5.30e-3 | 5.30e-3 | 5.30e-3 | 5.30e-3 | 182 | +2.90e-4 | +2.90e-4 | +2.90e-4 | -6.53e-4 |
| 156 | 3.00e-3 | 2 | 5.01e-3 | 5.25e-3 | 5.13e-3 | 5.01e-3 | 146 | -3.22e-4 | -5.30e-5 | -1.88e-4 | -5.66e-4 |
| 157 | 3.00e-3 | 2 | 5.01e-3 | 5.62e-3 | 5.32e-3 | 5.01e-3 | 146 | -7.92e-4 | +5.71e-4 | -1.11e-4 | -4.86e-4 |
| 158 | 3.00e-3 | 1 | 5.35e-3 | 5.35e-3 | 5.35e-3 | 5.35e-3 | 167 | +3.94e-4 | +3.94e-4 | +3.94e-4 | -3.98e-4 |
| 159 | 3.00e-3 | 2 | 5.01e-3 | 5.24e-3 | 5.12e-3 | 5.01e-3 | 146 | -3.14e-4 | -1.26e-4 | -2.20e-4 | -3.65e-4 |
| 160 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 177 | +4.03e-4 | +4.03e-4 | +4.03e-4 | -2.88e-4 |
| 161 | 3.00e-3 | 3 | 4.96e-3 | 5.45e-3 | 5.21e-3 | 4.96e-3 | 146 | -3.46e-4 | +7.37e-5 | -1.90e-4 | -2.66e-4 |
| 162 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 180 | +6.55e-4 | +6.55e-4 | +6.55e-4 | -1.74e-4 |
| 163 | 3.00e-3 | 2 | 4.92e-3 | 5.15e-3 | 5.04e-3 | 4.92e-3 | 130 | -5.50e-4 | -3.57e-4 | -4.53e-4 | -2.26e-4 |
| 164 | 3.00e-3 | 2 | 4.95e-3 | 5.39e-3 | 5.17e-3 | 4.95e-3 | 130 | -6.48e-4 | +5.30e-4 | -5.91e-5 | -2.00e-4 |
| 165 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 172 | +5.71e-4 | +5.71e-4 | +5.71e-4 | -1.23e-4 |
| 166 | 3.00e-3 | 3 | 4.82e-3 | 5.44e-3 | 5.08e-3 | 4.82e-3 | 130 | -6.41e-4 | -2.89e-5 | -3.09e-4 | -1.75e-4 |
| 167 | 3.00e-3 | 1 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 151 | +3.96e-4 | +3.96e-4 | +3.96e-4 | -1.18e-4 |
| 168 | 3.00e-3 | 2 | 5.14e-3 | 5.41e-3 | 5.28e-3 | 5.14e-3 | 130 | -3.95e-4 | +3.45e-4 | -2.54e-5 | -1.04e-4 |
| 169 | 3.00e-3 | 3 | 4.99e-3 | 5.38e-3 | 5.14e-3 | 5.04e-3 | 136 | -5.84e-4 | +2.93e-4 | -7.28e-5 | -9.76e-5 |
| 170 | 3.00e-3 | 1 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 154 | +2.57e-4 | +2.57e-4 | +2.57e-4 | -6.21e-5 |
| 171 | 3.00e-3 | 3 | 4.75e-3 | 5.36e-3 | 5.02e-3 | 4.75e-3 | 113 | -6.40e-4 | +1.58e-4 | -2.89e-4 | -1.29e-4 |
| 172 | 3.00e-3 | 2 | 4.78e-3 | 5.16e-3 | 4.97e-3 | 4.78e-3 | 113 | -6.74e-4 | +5.73e-4 | -5.03e-5 | -1.20e-4 |
| 173 | 3.00e-3 | 2 | 4.80e-3 | 5.15e-3 | 4.97e-3 | 4.80e-3 | 107 | -6.41e-4 | +5.10e-4 | -6.52e-5 | -1.15e-4 |
| 174 | 3.00e-3 | 2 | 4.72e-3 | 5.20e-3 | 4.96e-3 | 4.72e-3 | 101 | -9.74e-4 | +5.93e-4 | -1.90e-4 | -1.37e-4 |
| 175 | 3.00e-3 | 3 | 4.56e-3 | 4.99e-3 | 4.72e-3 | 4.61e-3 | 107 | -9.01e-4 | +4.27e-4 | -1.22e-4 | -1.36e-4 |
| 176 | 3.00e-3 | 2 | 4.68e-3 | 5.02e-3 | 4.85e-3 | 4.68e-3 | 107 | -6.71e-4 | +6.30e-4 | -2.08e-5 | -1.21e-4 |
| 177 | 3.00e-3 | 3 | 4.71e-3 | 5.09e-3 | 4.84e-3 | 4.71e-3 | 101 | -7.50e-4 | +6.16e-4 | -5.04e-5 | -1.07e-4 |
| 178 | 3.00e-3 | 2 | 4.52e-3 | 5.36e-3 | 4.94e-3 | 4.52e-3 | 92 | -1.84e-3 | +9.42e-4 | -4.50e-4 | -1.86e-4 |
| 179 | 3.00e-3 | 3 | 4.19e-3 | 5.36e-3 | 4.73e-3 | 4.19e-3 | 81 | -1.55e-3 | +1.21e-3 | -5.32e-4 | -3.03e-4 |
| 180 | 3.00e-3 | 4 | 3.98e-3 | 4.75e-3 | 4.24e-3 | 3.98e-3 | 75 | -1.92e-3 | +1.11e-3 | -2.98e-4 | -3.12e-4 |
| 181 | 3.00e-3 | 3 | 3.99e-3 | 4.51e-3 | 4.19e-3 | 3.99e-3 | 75 | -1.41e-3 | +1.19e-3 | -1.50e-4 | -2.81e-4 |
| 182 | 3.00e-3 | 3 | 3.92e-3 | 4.70e-3 | 4.21e-3 | 3.92e-3 | 75 | -2.12e-3 | +1.45e-3 | -3.29e-4 | -3.10e-4 |
| 183 | 3.00e-3 | 3 | 4.09e-3 | 4.85e-3 | 4.38e-3 | 4.20e-3 | 71 | -2.42e-3 | +1.95e-3 | -3.78e-5 | -2.50e-4 |
| 184 | 3.00e-3 | 2 | 4.07e-3 | 6.58e-3 | 5.32e-3 | 6.58e-3 | 263 | -4.36e-4 | +1.83e-3 | +6.97e-4 | -5.90e-5 |
| 185 | 3.00e-3 | 1 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 280 | +4.11e-4 | +4.11e-4 | +4.11e-4 | -1.20e-5 |
| 186 | 3.00e-3 | 1 | 7.52e-3 | 7.52e-3 | 7.52e-3 | 7.52e-3 | 285 | +6.50e-5 | +6.50e-5 | +6.50e-5 | -4.30e-6 |
| 188 | 3.00e-3 | 2 | 7.24e-3 | 7.53e-3 | 7.38e-3 | 7.24e-3 | 263 | -1.48e-4 | +2.88e-6 | -7.27e-5 | -1.80e-5 |
| 189 | 3.00e-3 | 1 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 266 | +2.97e-5 | +2.97e-5 | +2.97e-5 | -1.33e-5 |
| 191 | 3.00e-3 | 2 | 7.75e-3 | 8.02e-3 | 7.89e-3 | 7.75e-3 | 278 | -1.27e-4 | +2.76e-4 | +7.48e-5 | +1.44e-6 |
| 193 | 3.00e-3 | 2 | 7.79e-3 | 8.31e-3 | 8.05e-3 | 7.79e-3 | 284 | -2.24e-4 | +1.91e-4 | -1.69e-5 | -4.13e-6 |
| 195 | 3.00e-3 | 1 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 357 | +2.11e-4 | +2.11e-4 | +2.11e-4 | +1.74e-5 |
| 196 | 3.00e-3 | 1 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 267 | -2.98e-4 | -2.98e-4 | -2.98e-4 | -1.42e-5 |
| 197 | 3.00e-3 | 1 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 273 | -1.12e-4 | -1.12e-4 | -1.12e-4 | -2.40e-5 |
| 198 | 3.00e-3 | 1 | 7.55e-3 | 7.55e-3 | 7.55e-3 | 7.55e-3 | 271 | +1.22e-5 | +1.22e-5 | +1.22e-5 | -2.04e-5 |
| 199 | 3.00e-3 | 1 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 266 | -1.08e-5 | -1.08e-5 | -1.08e-5 | -1.94e-5 |

